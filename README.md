# CLIP-MMD
An unofficial implementation of [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). Uses Transformers on PyTorch.

### Features
- support multiple GPUs with DDP.
- acquire less memory; support larger dataset (like 1M each).
- uses resizer of PIL for more accurate results, recommended by [Clean-FID](https://github.com/GaParmar/clean-fid).
- support multiple input: folder, Dataset(for calculation during training), pre-computed stastics.


### Comparison
Measure time (second) of calculation. Cuda is on single V100.
|n, m, dim|original|efficient|(by row)|(cuda)|(cuda + by row)|
|-|-|-|-|-|-|
2048, 16384, 768|1.379|1.142|0.421|0.460|0.164|
16384, 16384, 768|3.312|2.764|1.137|0.567|0.391|
50000, 50000, 768|29.74|26.64|12.85|1.413|1.383|
1280000, 50000, 768|OOM|OOM|98.21|OOM|4.446|

### Thanks
To [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch). Some logics came from it.


### Usage
```
pip install clip-mmd
```

CLI interface
```
usage: clip-mmd [-h] [--no-cuda] [--gpus GPUS] [--no-mem-save] [--batch-size BATCH_SIZE] [--calculator-bs CALCULATOR_BS] [--model MODEL] [--num-workers NUM_WORKERS] [--extract-mode] [--size SIZE]
                [--interpolation {nearest,bilinear,bicubic,lanczos,hamming,box}]
                data_path_1 [data_path_2]

Command Line Interface for CMMD calculations and feature extraction.

positional arguments:
  data_path_1           Path to the first data folder or file
  data_path_2           Path to the second data folder or file, or output path of pre-extracted features

options:
  -h, --help            show this help message and exit
  --no-cuda             only use cpu
  --gpus GPUS           Comma-separated list of GPUs to use (e.g., 0,1,2,3). Use it if you want do on multi gpus.
  --no-mem-save         Flag to disable memory-saving features
  --batch-size BATCH_SIZE
                        Batch size for processing
  --calculator-bs CALCULATOR_BS
                        Batch size for the CMMD calculator
  --model MODEL         Model to use for feature extraction. Default: openai/clip-vit-large-patch14-336,
  --num-workers NUM_WORKERS
                        Numbers of dataloader workers
  --extract-mode        If enabled, only extract reatures from data_path_1, and save to data_path_2.
  --size SIZE           Image patch size for model input
  --interpolation {nearest,bilinear,bicubic,lanczos,hamming,box}
                        Interpolation algorithm for resampling an image. Default: bicubic.

```

example
```
clip-mmd /path/to/generated_samples.npz /path/to/reference/imgs/
clip-mmd /path/to/images/ features/for/them.pth --gpus 3,4,5,6 --batch-size 128 --extract-mode
clip-mmd /path/to/features1.pth /path/to/features2.pth
clip-mmd /folder/1  /folder/2 --interpolation bilinear
clip-mmd /pre/extracted/features.pth /sampled/images 
```

OR directly use it:

```python
from clip_mmd import logic
prep = logic.CMMD(data_parallel=True, device=[2,3,4,5])
distance = prep.execute('path/to/folder/1', 'path/to/folder/2')
print(f'CMMD: {distance:5f}')
```
You can also use path to sampled npz or pth, pre-caculated statics, or even customized Dataset.