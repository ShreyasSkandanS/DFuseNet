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

import pdb
import time

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

class FeatureExtractDepth(nn.Module):
    """
    Feature extraction block for Depth branch
    """
    def __init__(self):
        super(FeatureExtractDepth, self).__init__()
        self.inplanes = 32

        self.conv_block1 = nn.Sequential(ConvBN(1, 16, 11, 1, 5, 1),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(ConvBN(16, 32, 7, 2, 3, 1),
                                         nn.ReLU())
        self.conv_block3 = nn.Sequential(ConvBN(32, 64, 5, 2, 2, 1),
                                         nn.ReLU())

        self.level64_pool = nn.MaxPool2d((64, 64), stride=(64,64))
        self.level64_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level64_relu = nn.ReLU()

        self.level32_pool = nn.MaxPool2d((32, 32), stride=(32,32))
        self.level32_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level32_relu = nn.ReLU()

        self.level16_pool = nn.MaxPool2d((16, 16), stride=(16,16))
        self.level16_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level16_relu = nn.ReLU()

        self.level8_pool = nn.MaxPool2d((8, 8), stride=(8,8))
        self.level8_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level8_relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        m_in = (x > 0).detach().float()
        new_conv1 = self.conv_block1(x)
        new_conv2 = self.conv_block2(new_conv1)
        new_conv3 = self.conv_block3(new_conv2)
        interp_size = (new_conv3.size()[2], new_conv3.size()[3])
        op_maskconv = new_conv3

        op_l64_pool     = self.level64_pool(op_maskconv)
        op_l64_conv     = self.level64_conv(op_l64_pool)
        op_l64          = self.level64_relu(op_l64_conv)
        op_l64_upsample = F.interpolate(input = op_l64,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l32_pool     = self.level32_pool(op_maskconv)
        op_l32_conv     = self.level32_conv(op_l64_pool)
        op_l32          = self.level32_relu(op_l64_conv)
        op_l32_upsample = F.interpolate(input = op_l32,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l16_pool     = self.level16_pool(op_maskconv)
        op_l16_conv     = self.level16_conv(op_l16_pool)
        op_l16          = self.level16_relu(op_l16_conv)
        op_l16_upsample = F.interpolate(input = op_l16,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l8_pool      = self.level8_pool(op_maskconv)
        op_l8_conv      = self.level8_conv(op_l8_pool)
        op_l8           = self.level8_relu(op_l8_conv)
        op_l8_upsample = F.interpolate(input = op_l8,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')
        return op_maskconv, op_l8_upsample, op_l16_upsample,\
               op_l32_upsample, op_l64_upsample
