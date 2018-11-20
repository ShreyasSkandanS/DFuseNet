# ******************************************************************************
# 				DFuseNet
#
#    Shreyas S. Shivakumar, Ty Nguyen, Steven W.Chen and Camillo J. Taylor
#
# 				( 2018 )
#
# Code layout has been adapted from https://github.com/yxgeee/DepthComplete, for
# simplicity of usage with KITTI dataset. The rest of the code has been written
# by Shreyas S. Shivakumar and Ty Nguyen
#
#       University of Pennsylvania | {sshreyas,tynguyen}@seas.upenn.edu
#
# ******************************************************************************
from __future__ import print_function, absolute_import
import argparse
import os,sys
import shutil
import time
import math
import os.path as osp
import numpy as np
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import models
from models.DFuseNet import *
from preprocess import *


# ************************ ARGUMENTS FOR INFERENCE ******************************
parser = argparse.ArgumentParser(description='DFuseNet: Inference Parameters')
parser.add_argument('--model',
                    type    = str,
                    metavar = 'PATH',
                    default = 'pretrained_weights/best_model.pth.tar',
                    help    = 'Path to pretrained weights (default: none)')
parser.add_argument('--rgbin',
                    default = '../../val_selection_cropped/image/',
                    help    = 'Path to folder of RGB images')
parser.add_argument('--din',
                    default = '../../val_selection_cropped/groundtruth_depth/',
                    help    = 'Path to folder of Depth images')
parser.add_argument('--dout',
                    default = '../prediction/',
                    help    = 'Path to folder of Predictions images')
parser.add_argument('--gpuid',
                    default = '0',
                    type    = str,
                    help    = 'GPU device ids (CUDA_VISIBLE_DEVICES)')
# ******************************************************************************

def main():
    global args
    args = parser.parse_args()

    # Set the GPU device IDs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    cudnn.benchmark = True

    # Create folder to save prediction images
    save_root = args.dout
    if not osp.isdir(save_root):
        os.makedirs(save_root)

    # Initialize model and display total parameters
    model = models.init_model(name='dfusenet')
    print("Total parameters: {:.5f}M".format(
        sum(p.numel() for p in model.parameters())/1000000.0))

    # Load pre-trained weights
    if os.path.isfile(args.model):
        print("Loading pre-trained weights '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("No model found, please check location and try again '{}'".format(args.resume))
        return

    model = torch.nn.DataParallel(model).cuda()
    tensor_to_pil = T.ToPILImage()
    image_folder = sorted(glob.glob(args.rgbin + '/*.png'))
    laser_folder = sorted(glob.glob(args.din + '/*.png'))
    model.eval()

    print("Beginning evaluation..")

    with torch.no_grad():
        for img_idx in range(0,len(laser_folder)):

            # Iterate through input folder contents
            laser_path  = laser_folder[img_idx]
            left_path   = image_folder[img_idx]
            print("Currently processing the following images..")
            print("RGB Image: {}".format(left_path))
            print("D   Image: {}".format(laser_path))

            # Load and pre-process input lidar data
            laser_pil   = Image.open(laser_path)
            laser       = in_depth_transform(laser_pil)

            # Load and pre-process input RGB data
            left_pil    = Image.open(left_path)
            left        = color_transform(left_pil)

            # Convert RGB and lidar data to Tensor
            left        = TF.to_tensor(left).float()
            laser       = TF.to_tensor(laser).float()
            laser       = torch.unsqueeze(laser,0).cuda()
            left        = torch.unsqueeze(left,0).cuda()

            # We stack the input and rgb tensors as one tensor for simplicity
            input_stack = torch.cat((left, laser), dim=1)

            # Perform forward pass through the network
            output      = model(input_stack)

            # Rescale data back into original KITTI format
            output      = output * 256.
            laser       = laser * 256.

            output      = output[0].cpu()
            pil_img     = tensor_to_pil(output.int())
            filename    = laser_folder[img_idx].split('/')[-1]
            pil_img.save(save_root + filename)
            print('Finished processing: {}'.format(filename))

if __name__ == '__main__':
    main()
