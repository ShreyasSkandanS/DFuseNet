# ******************************************************************************
#                               DFuseNet
#
#    Shreyas S. Shivakumar, Ty Nguyen, Steven W.Chen and Camillo J. Taylor
#
#                               ( 2018 )
#
# Some code has been adapted from "https://github.com/kujason/ip_basic", for
# simplicity of usage with KITTI dataset. The rest of the code has been written
# by Shreyas S. Shivakumar and Ty Nguyen
#
#       University of Pennsylvania | {sshreyas,tynguyen}@seas.upenn.edu
#
# ******************************************************************************
from __future__ import print_function, absolute_import
import os
import numpy as np
import os.path as osp
import cv2
import pdb

# KERNELS for depth filling
FULL_KERNEL_5   = np.ones((5, 5), np.uint8)
FULL_KERNEL_7   = np.ones((7, 7), np.uint8)
FULL_KERNEL_31  = np.ones((31, 31), np.uint8)
DIAMOND_KERNEL_5 = np.array([\
        [0, 0, 1, 0, 0],\
        [0, 1, 1, 1, 0],\
        [1, 1, 1, 1, 1],\
        [0, 1, 1, 1, 0],\
        [0, 0, 1, 0, 0],\
    ], dtype=np.uint8)

def fill_in_fast(depth_map,
                 max_depth      = 100.0,
                 custom_kernel  = DIAMOND_KERNEL_5):
    """
    This function was adapted from "https://github.com/kujason/ip_basic"
    written by Jason Ku, Ali Harakeh and Steven L. Waslander.
    Please make necessary citations if you use this code.
    """
    depth_map               = np.float32(depth_map)
    valid_pixels            = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    depth_map               = cv2.dilate(depth_map, custom_kernel)
    depth_map               = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)
    empty_pixels            = (depth_map < 0.1)
    dilated                 = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    top_row_pixels          = np.argmax(depth_map > 0.1, axis=0)
    top_pixel_values        = depth_map[top_row_pixels, range(depth_map.shape[1])]
    for pixel_col_idx in range(depth_map.shape[1]):
        depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]
    empty_pixels            = depth_map < 0.1
    dilated                 = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]
    depth_map               = cv2.medianBlur(depth_map, 5)

    valid_pixels            = (depth_map > 0.1)
    blurred                 = cv2.GaussianBlur(depth_map, (5, 5), 0)
    depth_map[valid_pixels] = blurred[valid_pixels]
    empty_pixels            = depth_map < 0.1
    dilated                 = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    depth_map               = cv2.medianBlur(depth_map, 5)
    valid_pixels            = (depth_map > 0.1)
    blurred                 = cv2.GaussianBlur(depth_map, (5, 5), 0)
    depth_map[valid_pixels] = blurred[valid_pixels]
    valid_pixels            = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
    depth_map               = np.expand_dims(depth_map, axis=2)

    return depth_map

def in_depth_transform(pil_img):
    """
    Function to properly scale and handle KITTI depth data
    """
    depth_png = np.array(pil_img, dtype = int)[:,:,np.newaxis]
    assert(np.max(depth_png) > 255)
    # Scale KITTI depth to correct depth value
    depth_png = depth_png.astype(np.float) / 256.
    # Note: we set a maximum depth value for KITTI as 100m
    depth = fill_in_fast(depth_png,
                         max_depth      = 100.0,
                         custom_kernel  = DIAMOND_KERNEL_5)
    depth[depth == 0] = -1.
    return depth

def color_transform(left_pil):
    """
    # == for kitti selected validation set
    Function to properly scale and handle KITTI color image data
    """
    left_png    = np.array(left_pil, dtype=int)
    assert(np.max(left_png) < 256 + 1)
    left        = left_png.astype(np.float) / 255.
    return left

