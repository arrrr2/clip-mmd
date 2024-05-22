# efficient-CMMD-pytorch
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
To [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch). Main logics came from it.