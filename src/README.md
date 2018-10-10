# SparseNet implementation

## CLAIM
The original result in paper was produced by 'pytorch/denselink.py'. 
However, because of tensor contiguous bugs, the performance can only be repliacted under CUDA 7.5 and CUDNN 5. 
To avoid this issue, we now provide alternative TensorFlow implementation. Note the performance is slightly worse than what claimed in report.

## Pretrained
The pretrained models are avaliable at [Google Drive](https://drive.google.com/drive/folders/1UXEZKoPuN-PpqGUtDBBO2Bz7YYLodLNP?usp=sharing).