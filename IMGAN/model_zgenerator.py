import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import cv2
import mcubes

from ops import *

class ZGenerator(object):
	def __init__(self, sess, real_size, batch_size_input, is_training = False, z_dim=128, ef_dim=32, gf_dim=128, dataset_name='default', z_vectors='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16*2)
		#3-- (64, 32*32*32)
		#4-- (128, 32*32*32*4)
		self.real_size = real_size #output point-value voxel grid size in training
		self.batch_size_input = batch_size_input #training batch size (virtual, batch_size is the real batch_size)
		
		self.batch_size = 16*16*16*4 #adjust batch_size according to gpu memory size in training
		if self.batch_size_input<self.batch_size:
			self.batch_size = self.batch_size_input
		
		self.input_size = 64 #input voxel grid size

		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir

		if os.path.exists(self.data_dir+'/'+self.dataset_name+'.hdf5'):
			self.data_dict = h5py.File(self.data_dir+'/'+self.dataset_name+'.hdf5', 'r')
			self.data_points = self.data_dict['points_'+str(self.real_size)][:]
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_voxels = self.data_dict['voxels'][:]
			if self.batch_size_input!=self.data_points.shape[1]:
				print("error: batch_size!=data_points.shape")
				exit(0)
			if self.input_size!=self.data_voxels.shape[1]:
				print("error: input_size!=data_voxels.shape")
				exit(0)
		else:
			if is_training:
				print("error: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				exit(0)
			else:
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')

		self.z_vectors = pd.read_csv('data/' + z_vectors + '.csv', header = None)
		print(self.z_vectors.shape)
		
		if not is_training:
			self.real_size = 64 #output point-value voxel grid size in testing
			self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
			self.batch_size = self.test_size*self.test_size*self.test_size #do not change
			
			#get coords
			dima = self.test_size
			dim = self.real_size
			self.aux_x = np.zeros([dima,dima,dima],np.uint8)
			self.aux_y = np.zeros([dima,dima,dima],np.uint8)
			self.aux_z = np.zeros([dima,dima,dima],np.uint8)
			multiplier = int(dim/dima)
			multiplier2 = multiplier*multiplier
			multiplier3 = multiplier*multiplier*multiplier
			for i in range(dima):
				for j in range(dima):
					for k in range(dima):
						self.aux_x[i,j,k] = i*multiplier
						self.aux_y[i,j,k] = j*multiplier
						self.aux_z[i,j,k] = k*multiplier
			self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
			self.coords = (self.coords+0.5)/dim*2.0-1.0
			self.coords = np.reshape(self.coords,[multiplier3,self.batch_size,3])
		
		self.build_model()

	def build_model(self):
		self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32)
		self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)
		self.point_value = tf.placeholder(shape=[self.batch_size,1], dtype=tf.float32)
		
		self.zG = self.generator(self.point_coord, self.z_vector, phase_train=True, reuse=False)
		
		self.loss = tf.reduce_mean(tf.square(self.point_value - self.zG))
		
		self.saver = tf.train.Saver(max_to_keep=10)
		
		
	def generator(self, points, z, phase_train=True, reuse=False):
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			zs = tf.tile(z, [self.batch_size,1])
			pointz = tf.concat([points,zs],1)
			print("pointz",pointz.shape)
			
			h1 = lrelu(linear(pointz, self.gf_dim*16, 'h1_lin'))
			h1 = tf.concat([h1,pointz],1)
			
			h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin'))
			h2 = tf.concat([h2,pointz],1)
			
			h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin'))
			h3 = tf.concat([h3,pointz],1)
			
			h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin'))
			h4 = tf.concat([h4,pointz],1)
			
			h5 = lrelu(linear(h4, self.gf_dim, 'h7_lin'))
			h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))
			
			return tf.reshape(h6, [self.batch_size,1])
	
	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		self.sess.run(tf.global_variables_initializer())
		
		batch_idxs = len(self.data_points)
		batch_index_list = np.arange(batch_idxs)
		batch_num = int(self.batch_size_input/self.batch_size)
		if self.batch_size_input%self.batch_size != 0:
			print("batch_size_input % batch_size != 0")
			exit(0)
		
		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in range(counter, config.epoch):
			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_num = 0
			for idx in range(0, batch_idxs):
				for minib in range(batch_num):
					dxb = batch_index_list[idx]
					# batch_voxels = self.data_voxels[dxb:dxb+1]
					batch_z_vector = self.z_vectors[dxb:dxb+1]
					batch_points_int = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
					batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
					batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
					
					# Update AE network
					_, errAE = self.sess.run([ae_optim, self.loss],
						feed_dict={
							self.z_vector: batch_z_vector,
							self.point_coord: batch_points,
							self.point_value: batch_values,
						})
					avg_loss += errAE
					avg_num += 1
					if (idx%16 == 0):
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, avgloss: %.8f" % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, errAE, avg_loss/avg_num))

				if idx==batch_idxs-1:
					model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
					real_model_float = np.zeros([self.real_size,self.real_size,self.real_size],np.float32)
					for minib in range(batch_num):
						dxb = batch_index_list[idx]
						# batch_voxels = self.data_voxels[dxb:dxb+1]
						batch_z_vector = self.z_vectors[dxb:dxb+1]
						batch_points_int = self.data_points[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						batch_points = (batch_points_int+0.5)/self.real_size*2.0-1.0
						batch_values = self.data_values[dxb,minib*self.batch_size:(minib+1)*self.batch_size]
						
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z_vector,
								self.point_coord: batch_points,
							})
						model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size])
						real_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])
					img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1t.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2t.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3t.png",img3)
					img1 = np.clip(np.amax(real_model_float, axis=0)*256, 0,255).astype(np.uint8)
					img2 = np.clip(np.amax(real_model_float, axis=1)*256, 0,255).astype(np.uint8)
					img3 = np.clip(np.amax(real_model_float, axis=2)*256, 0,255).astype(np.uint8)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_1i.png",img1)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_2i.png",img2)
					cv2.imwrite(config.sample_dir+"/"+str(epoch)+"_3i.png",img3)
					print("[sample]")
				
				if idx==batch_idxs-1:
					self.save(config.checkpoint_dir, epoch)
	
	def test(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		
		for t in range(self.data_voxels.shape[0]):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						# batch_voxels = self.data_voxels[t:t+1]
						batch_z_vector = self.z_vectors[t:t+1]
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z_vector,
								self.point_coord: self.coords[minib],
							})
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			
			thres = 0.5
			vertices, triangles = mcubes.marching_cubes(model_float, thres)
			mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+"out"+str(t)+".dae", str(t))
			
			print("[sample]")
	
	def test_z(self, config, batch_z, dim=64):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 256
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])
		
		for t in range(batch_z.shape[0]):
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						print(t,i,j,k)
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
			
			thres = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
			for i in range(len(thres)):
				vertices, triangles = mcubes.marching_cubes(model_float, thres[i])
				mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+str(t)+"_"+str(thres[i]).replace(".", "")+".dae", str(t))
			
			print("[sample GAN]")
	
	def test_z_variations(self, config, batch_z, dim=64):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 256
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])
		
		for t in range(batch_z.shape[0]):
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						print(t,i,j,k)
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t//7+1)+"_"+str(t%7+1)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t//7+1)+"_"+str(t%7+1)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t//7+1)+"_"+str(t%7+1)+"_3t.png",img3)
			
			thres = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
			for i in range(len(thres)):
				vertices, triangles = mcubes.marching_cubes(model_float, thres[i])
				mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+str(t//7+1)+"_"+str(t%7+1)+"_"+str(thres[i]).replace(".", "")+".dae", str(t))
			
			print("[sample GAN]")
	
	def test_mu_sigma(self, config, batch_mu, batch_sigma, dim=64, sample_num=100):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		dima = self.test_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		
		#get coords 256
		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		coords = (coords+0.5)/dim*2.0-1.0
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])

		batch_z = np.random.normal(batch_mu, batch_sigma, (sample_num, batch_mu.shape[0], batch_mu.shape[1]))
		batch_z = batch_z.reshape((-1, batch_mu.shape[1]))
		
		for t in range(batch_z.shape[0]):
			model_float = np.zeros([dim+2,dim+2,dim+2],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						print(t,i,j,k)
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.zG,
							feed_dict={
								self.z_vector: batch_z[t:t+1],
								self.point_coord: coords[minib],
							})
						model_float[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out, [dima,dima,dima])
			img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			cv2.imwrite(config.sample_dir+"/"+str(t%batch_mu.shape[0]+1)+"_"+str(t//batch_mu.shape[0]+1)+"_1t.png",img1)
			cv2.imwrite(config.sample_dir+"/"+str(t%batch_mu.shape[0]+1)+"_"+str(t//batch_mu.shape[0]+1)+"_2t.png",img2)
			cv2.imwrite(config.sample_dir+"/"+str(t%batch_mu.shape[0]+1)+"_"+str(t//batch_mu.shape[0]+1)+"_3t.png",img3)
			
			thres = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
			for i in range(len(thres)):
				vertices, triangles = mcubes.marching_cubes(model_float, thres[i])
				mcubes.export_mesh(vertices, triangles, config.sample_dir+"/"+str(t%batch_mu.shape[0]+1)+"_"+str(t//batch_mu.shape[0]+1)+"_"+str(thres[i]).replace(".", "")+".dae", str(t))
			
			print("[sample GAN]")

	@property
	def model_dir(self):
		return "{}_{}_{}".format(
				self.dataset_name, self.input_size, self.z_dim)
			
	def save(self, checkpoint_dir, step):
		model_name = "ZGenerator.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
