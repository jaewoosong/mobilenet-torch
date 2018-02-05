# mobilenet-torch
Torch implementation of MobileNet (https://arxiv.org/abs/1704.04861), implemented by Jaewoo Song.

# CUDNN and NN versions

## Why two versions?
* MobileNet uses __depthwise separable convolutions__. They are implemented differently in Torch.
* Only `nn` module has function named `nn.SpatialDepthWiseConvolution()`.
* On the other hand, `cudnn` module's function `cudnn.SpatialConvolution()` has one more parameter than its equivalent in `nn`. By setting `nInputPlane`, the very last parameter, one can implement depthwise separable convolutions.

## So, any difference?
* It is much faster to train the `cudnn` version. Training the `nn` version is really slow.
* But under the circumstances without GPUs, `nn` version is the only choice.
