# ******************************************************************************
#                               DFuseNet
#
#    Shreyas S. Shivakumar, Ty Nguyen, Steven W.Chen and Camillo J. Taylor
#
#                               ( 2018 )
#
#       This code has been written Shreyas S. Shivakumar and Ty Nguyen
#
#       University of Pennsylvania | {sshreyas,tynguyen}@seas.upenn.edu
#
# ******************************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

def ConvBN(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """
    Perform 2D Convolution with Batch Normalization
    """
    return nn.Sequential(nn.Conv2d(in_planes,
                                   out_planes,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = dilation if dilation > 1 else pad,
                                   dilation = dilation,
                                   bias=False),
                         nn.BatchNorm2d(out_planes))

class BasicResBlock(nn.Module):
    """
    Basic Convolution block with Residual
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicResBlock, self).__init__()
        self.conv1 = ConvBN(inplanes, planes, 3, stride, pad, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBN(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

class FeatureExtractRGB(nn.Module):
    """
    Feature extraction block for RGB branch
    """
    def __init__(self):
        super(FeatureExtractRGB, self).__init__()
        self.inplanes = 32

        self.fe_conv1 = ConvBN(3,32,3,2,1,1)
        self.fe_relu1 = nn.ReLU()
        self.fe_conv2 = ConvBN(32,32,3,1,1,1)
        self.fe_relu2 = nn.ReLU()

        self.fe_conv3_4   = self._make_layer(BasicResBlock,32,3,1,1,1)
        self.fe_conv5_8   = self._make_layer(BasicResBlock,64,16,2,1,1)
        self.fe_conv9_10  = self._make_layer(BasicResBlock,128,3,1,1,1)
        self.fe_conv11_12 = self._make_layer(BasicResBlock,128,3,1,1,2)

        self.level64_pool = nn.AvgPool2d((64, 64), stride=(64,64))
        self.level64_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level64_relu = nn.ReLU()

        self.level32_pool = nn.AvgPool2d((32, 32), stride=(32,32))
        self.level32_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level32_relu = nn.ReLU()

        self.level16_pool = nn.AvgPool2d((16, 16), stride=(16,16))
        self.level16_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level16_relu = nn.ReLU()

        self.level8_pool = nn.AvgPool2d((8, 8), stride=(8,8))
        self.level8_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level8_relu = nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            pad,
                            dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        op_conv1 = self.fe_conv1(x)
        op_relu1 = self.fe_relu1(op_conv1)
        op_conv2 = self.fe_conv2(op_relu1)
        op_relu2 = self.fe_relu2(op_conv2)

        op_conv3_4   = self.fe_conv3_4(op_relu2)
        op_conv5_8   = self.fe_conv5_8(op_conv3_4)
        op_conv9_10  = self.fe_conv9_10(op_conv5_8)
        op_conv11_12 = self.fe_conv11_12(op_conv9_10)

        interp_size = (op_conv11_12.size()[2], op_conv11_12.size()[3])

        op_l64_pool     = self.level64_pool(op_conv11_12)
        op_l64_conv     = self.level64_conv(op_l64_pool)
        op_l64          = self.level64_relu(op_l64_conv)
        op_l64_upsample = F.interpolate(input = op_l64,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l32_pool     = self.level32_pool(op_conv11_12)
        op_l32_conv     = self.level32_conv(op_l64_pool)
        op_l32          = self.level32_relu(op_l64_conv)
        op_l32_upsample = F.interpolate(input = op_l32,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l16_pool     = self.level16_pool(op_conv11_12)
        op_l16_conv     = self.level16_conv(op_l16_pool)
        op_l16          = self.level16_relu(op_l16_conv)
        op_l16_upsample = F.interpolate(input = op_l16,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l8_pool      = self.level8_pool(op_conv11_12)
        op_l8_conv      = self.level8_conv(op_l8_pool)
        op_l8           = self.level8_relu(op_l8_conv)
        op_l8_upsample = F.interpolate(input = op_l8,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        return op_conv5_8, op_conv11_12, op_l8_upsample, op_l16_upsample,\
               op_l32_upsample, op_l64_upsample
