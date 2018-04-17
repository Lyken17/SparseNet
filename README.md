# SparseNet
Sparsely Aggregated Convolutional Networks [[PDF](https://arxiv.org/abs/1801.05895)]

[Ligeng Zhu](https://lzhu.me), [Ruizhi Deng](http://www.sfu.ca/~ruizhid/), [Michael Maire](http://ttic.uchicago.edu/~mmaire/), [Zhiwei Deng](http://www.sfu.ca/~zhiweid/), [Greg Mori](http://www.cs.sfu.ca/~mori/), [Ping Tan](https://www.cs.sfu.ca/~pingtan/)

# What is SparseNet?
SparseNet is a  network architecture that only aggregates previous layers with exponential offset, for example, i - 1, i - 2, i - 4, i - 8, i - 16 ... 

![](images/dense_and_sparse.png)

# Why use SparseNet?
The connectivity pattern yields state-of-the-art arruacies on small dataset CIFAR/10/100. On large scale ILSVRC 2012 (ImageNet) dataset, SparseNet achieves similar accuracy as ResNet and DenseNet, while only using much less parameters. 

# Better Performance

<table>
<tr><th> Without BC </th><th> With BC </th></tr>
<tr><td>

Architecture | Params | CIFAR 100
--- | --- | ---
DenseNet-40-12  | 1.1M | 24.79
DenseNet-100-12 | 7.2M | 20.97
DenseNet-100-24 | 28.28M | 19.61
--- | --- | ---
SparseNet-40-24  | 0.76M | 24.65
SparseNet-100-36 | 5.65M | 20.50
SparseNet-100-{16,32,64} | 7.22M | 19.49


</td><td>

Architecture | Params | CIFAR 100
--- | --- | ---
DenseNet-100-12 | 0.8M | 22.62
DenseNet-250-24 | 15.3M | 17,6
DenseNet-190-40 | 25.6M | 17.53
--- | --- | ---
SparseNet-100-24  | 1.46M | 22.12
SparseNet-100-{16,32,64} | 4.38M | 19.71
SparseNet-100-{32,64,128} | 16.72M | 17.71


</td></tr> </table>


## Efficient Parameter Utilization
* Parameter efficiency on CIFAR
  ![](images/cropped_two-weights-int.jpg)

* Paramter efficiency on ImageNet

  We notice sparsenet shows comparable efficiency even compared with pruned models.
  ![](images/imagenet_efficiency.png)
  
  
# Pretrained model
Refer for [source folder](src/).

# Cite
If SparseNet helps your research, please cite our work :) 

```
@article{DBLP:journals/corr/abs-1801-05895,
  author    = {Ligeng Zhu and
               Ruizhi Deng and
               Michael Maire and
               Zhiwei Deng and
               Greg Mori and
               Ping Tan},
  title     = {Sparsely Aggregated Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1801.05895},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.05895},
  archivePrefix = {arXiv},
  eprint    = {1801.05895},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-05895},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
