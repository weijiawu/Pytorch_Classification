## Introduction

Image classification based on PyTorch

## Major features

- Various backbones and pretrained models
- Large-scale training configs
- High efficiency and extensibility

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101

## Dataset
Supported:
- [x] CIFAR10
- [x] cifar10
- [x] ImageNet

## Benchmark 

Supported backbones:
- [x] VGG
- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] Densenet
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Efficientnet b0-b7

## loss function

## Train

```bash
# use gpu to train vgg16
$ python Train_CIFAR.py -Backbone vgg16 -Datasets cifar10
```

## Test 
Test the model using eval.py
```bash
$ python eval.py -Backbone vgg16 -resume path_to_vgg16_weights_file
```
## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- googlenet [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
- shufflenetv2 [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv2 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- residual attention network [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)
- senet [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- squeezenet [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4)
- nasnet [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012v4)


## Results
All models are trained in the same condition, and might not get the best result

|dataset|network|Params(M)|Flops(M)|Top-1(%)|Top-5(%)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar10|vgg11|128.80|7.613|90.02|-|
|cifar10|vgg11_bn|128.80|272.520|90.22|-|
|cifar10|ResNet-18|37.119|11.182|92.1|-|
|cifar10|ResNet-32|75.426|21.798|93.75|-|
|cifar10|ResNet-50|83.889|23.529|93.14|-|
|cifar10|ResNeXt29(32x4d)|4.774|779.626|93.76|-|
|cifar10|ResNeXt29(2x64d)|9.129|1417|93.83|-|
|cifar10|densenet121|7.979|2866|94.29|-|
|cifar10|mnasnet0_5|0.927|109.320|90.1|-|
|cifar10|shufflenet_v2_x0_5|0.344|0.817|91.0|-|
|cifar10|shufflenet_v2_x1_0|1.264|3.026|92.74|-|
|cifar10|shufflenet_v2_x1_5|2.489|300.281|92.83|-|
|cifar10|MobileNet_v2|6.398|2.237|93.28|-|
|cifar10| Efficientnet-b0|0.054|0.284|94.36|-|
|cifar10| Efficientnet-b1|0.073|0.387|94.68|-|
|cifar10| Efficientnet-b2|0.079|19.669|94.52|-|
|cifar10| Efficientnet-b3|0.102|26.065|-|-|
|:---:|:---:|:---:|:---:|:---:|:---:|
|cifar100| Efficientnet-b0|0.054|0.284|76.77|-|
|cifar100| Efficientnet-b1|0.073|0.387|76.420|-|
|cifar100| Efficientnet-b2|0.079|19.669|77.25|-|
|:---:|:---:|:---:|:---:|:---:|:---:|
|ImageNet| Efficientnet-b0|1.343|1.664|73.194|-|

More experiment results coming soon

## Contact

Eamil: wwj123@zju.edu.cn
