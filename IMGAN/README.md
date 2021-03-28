# Shape Decoder Network

## Training

### Chairs

Run the following commands.

```bash
python main.py --model zgenerator --train --epoch 50 --dataset chairs_vox --z_vectors chairs_z --real_size 16 --batch_size_input 4096 --sample_dir chairs_samples_train
python main.py --model zgenerator --train --epoch 100 --dataset chairs_vox --z_vectors chairs_z --real_size 32 --batch_size_input 8192 --sample_dir chairs_samples_train
python main.py --model zgenerator --train --epoch 200 --dataset chairs_vox --z_vectors chairs_z --real_size 64 --batch_size_input 32768 --sample_dir chairs_samples_train
```

### Lamps

Create a folder named `lamps_vox_64_128` under the folder `checkpoint`, copy all the checkpoints under `checkpoint/chairs_vox_64_128` to `checkpoint/lamps_vox_64_128`, and then run the following commands.

```bash
python main.py --model zgenerator --train --epoch 250 --dataset lamps_vox --z_vectors lamps_z --real_size 16 --batch_size_input 4096 --sample_dir lamps_samples_train
python main.py --model zgenerator --train --epoch 300 --dataset lamps_vox --z_vectors lamps_z --real_size 32 --batch_size_input 8192 --sample_dir lamps_samples_train
python main.py --model zgenerator --train --epoch 400 --dataset lamps_vox --z_vectors lamps_z --real_size 64 --batch_size_input 32768 --sample_dir lamps_samples_train
```

### Tables

Create a folder named `tables_vox_64_128` under the folder `checkpoint`, copy all the checkpoints under `checkpoint/chairs_vox_64_128` to `checkpoint/tables_vox_64_128`, and then run the following commands.

```bash
python main.py --model zgenerator --train --epoch 250 --dataset tables_vox --z_vectors tables_z --real_size 16 --batch_size_input 4096 --sample_dir tables_samples_train
python main.py --model zgenerator --train --epoch 300 --dataset tables_vox --z_vectors tables_z --real_size 32 --batch_size_input 8192 --sample_dir tables_samples_train
python main.py --model zgenerator --train --epoch 400 --dataset tables_vox --z_vectors tables_z --real_size 64 --batch_size_input 32768 --sample_dir tables_samples_train
```

## Generating Shapes from Gaussians

Run the following commands to generate shapes according to the example mu and sigma pairs prepared in the `data` folder.

```bash
python main.py --model zgenerator --dataset chairs_vox --mu_vectors chairs_mu --sigma_vectors chairs_sigma --sample_dir chairs_samples_gaussians
python main.py --model zgenerator --dataset lamps_vox --mu_vectors lamps_mu --sigma_vectors lamps_sigma --sample_dir lamps_samples_gaussians
python main.py --model zgenerator --dataset tables_vox --mu_vectors tables_mu --sigma_vectors tables_sigma --sample_dir tables_samples_gaussians
```
