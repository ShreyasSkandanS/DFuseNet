# ******************************************************************************
#                               DFuseNet
#
#    Shreyas S. Shivakumar, Ty Nguyen, Steven W.Chen and Camillo J. Taylor
#
#                               ( 2018 )
#
# 	This code has been written Shreyas S. Shivakumar and Ty Nguyen
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

from .fex_rgb import FeatureExtractRGB
from .fex_depth import FeatureExtractDepth

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
                                   bias=True),
                         nn.BatchNorm2d(out_planes))

class DFuseNet(nn.Module):
    def __init__(self, \
                 output_size=(352, 1216), \
                 in_channels=3):
        super(DFuseNet, self).__init__()
        self.KITTI_MAX_DEPTH = 100.0
        self.FeatureExtractRGB = FeatureExtractRGB()
        self.FeatureExtractDepth = FeatureExtractDepth()

        # Joint RGB and Depth layers
        self.joint_conv1 = nn.Sequential(ConvBN(320, 192, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(192, 192, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True),
                                       nn.ReLU())

        self.joint_conv2 = nn.Sequential(ConvBN(192, 192, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(192, 192, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True),
                                       nn.ReLU())

        self.joint_conv3 = nn.Sequential(ConvBN(192, 128, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))

        # Deconvolution / Reconstruction Layers
        self.deconv_l1 = nn.ConvTranspose2d(128,
                                            128,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            dilation = 1)
        self.deconv_l2 = nn.Sequential(ConvBN(128, 64, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l3 = nn.ConvTranspose2d(64,
                                            64,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1)
        self.deconv_l4 = nn.Sequential(ConvBN(64, 32, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l5 = nn.ConvTranspose2d(32,
                                            32,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            dilation = 1)
        self.deconv_l6 = nn.Sequential(ConvBN(32, 16, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(16, 16, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l7 = nn.ConvTranspose2d(16,
                                            16,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1)
        self.deconv_l8 = nn.Sequential(ConvBN(16, 1, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(1, 1, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))

        self.d_s1_depth = nn.Sequential(ConvBN(128, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                            padding=0, stride = 1,
                                                  bias = True))

        self.d_s2_depth = nn.Sequential(ConvBN(64, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                                  padding=0, stride = 1,
                                                  bias = True))


        self.d_s3_depth = nn.Sequential(ConvBN(32, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                                  padding=0, stride = 1,
                                                  bias = True))

        self.final_conv = nn.Sequential(ConvBN(113, 113, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(113, 1, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x_in):

        rgb_in = x_in[:,0:3,:,:]
        d_in = x_in[:,3,:,:] / self.KITTI_MAX_DEPTH
        i_raw, i_skip, i_b4, i_b3, \
        i_b2, i_b1 = self.FeatureExtractRGB(rgb_in)

        d_skip, d_b4, d_b3, \
        d_b2, d_b1 = self.FeatureExtractDepth(d_in)

        id_raw_cat = torch.cat((i_raw,
                                d_b4,
                                i_b4,
                                d_b3,
                                i_b3,
                                d_b2,
                                i_b2,
                                d_b1,
                                i_b1), 1)

        jf_s1 = self.joint_conv1(id_raw_cat)
        jf_s2 = self.joint_conv2(jf_s1)
        jf_s3 = self.joint_conv3(jf_s2)

        decon_1 = self.deconv_l1(jf_s3, output_size=(x_in.size()[2]//2,
                                                     x_in.size()[3]//2))
        decon_2 = self.deconv_l2(decon_1)
        decon_3 = self.deconv_l3(decon_2, output_size=(x_in.size()[2]//2,
                                                       x_in.size()[3]//2))
        decon_4 = self.deconv_l4(decon_3)
        decon_5 = self.deconv_l5(decon_4, output_size=(x_in.size()[2],
                                                       x_in.size()[3]))
        decon_6 = self.deconv_l6(decon_5)
        decon_7 = self.deconv_l7(decon_6, output_size=(x_in.size()[2],
                                                       x_in.size()[3]))
        decon_8 = self.deconv_l8(decon_7)

        decon_2_l1 = F.interpolate(input = decon_2,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_4_l1 = F.interpolate(input = decon_4,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None)
        decon_6_l1 = F.interpolate(input = decon_6,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_8_l1 = F.interpolate(input = decon_8,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_stack = torch.cat((decon_2_l1,
                                 decon_4_l1,
                                 decon_6_l1,
                                 decon_8_l1), 1)
        final_conv = self.final_conv(decon_stack)
        final_conv = self.final_sigmoid(final_conv) * self.KITTI_MAX_DEPTH
        return final_conv
