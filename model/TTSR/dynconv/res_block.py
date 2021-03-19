"""
    Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference
    https://github.com/thomasverelst/dynconv/blob/master/classification/models/resnet_util.py
"""

import sys
import pathlib

import torch
import torch.nn as nn

DIR_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(DIR_PATH))
import dynconv


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(
        self
        , inplanes
        , planes
        , stride=1
        , downsample=None
        , groups=1
        , base_width=64
        , dilation=1
        , sparse=True
        , scaling=None
    ):

        super().__init__()
        assert groups == 1
        assert dilation == 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.sparse = sparse

        if sparse:
            # in the resnet basic block, the first convolution is already strided, so mask_stride = 1
            self.masker = dynconv.MaskUnit(channels=inplanes, stride=stride, dilate_stride=1, scaling=scaling)

        self.fast = False  # :((

    def forward(self, input):
        x, meta = input
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.sparse:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity
        else:
            assert meta is not None
            m = self.masker(x, meta)
            mask_dilate, mask = m['dilate'], m['std']

            x = dynconv.conv3x3(self.conv1, x, None, mask_dilate)
            x = dyconv.relu(self.relu, x, mask_dilate)

            x = dynconv.conv3x3(self.conv2, x, mask_dilate, mask)
            x = dynconv.relu(None, x, mask)  # No relu applied actually (calling just for metainfo)

            out = identity + dynconv.apply_mask(x, mask)

        # out = self.relu(out)

        return out, meta
