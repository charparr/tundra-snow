# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:45:25 2016

@author: cparr
"""

# import the necessary packages
import os
from skimage import io
from skimage.util import random_noise
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def block_mean(ar, fact):
    
    # function to downsample inputs by a factor of two

    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


def convolve(image, kernel):
    
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
    
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
 
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
 
	pad = ( kW - 1 ) / 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
 
	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
 
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
      
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
      
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
 
			k = (roi * kernel).sum()
 
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
 
			output[y - pad, x - pad] = k
	return output
 
def input_data(path, filename, blur_amount):
    
    img_path = os.path.join(path, filename)
    img = io.imread(img_path)
    img = img[85:341,90:346]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(blur_amount,blur_amount),0)
    
    noise = random_noise(gray, mode = "gaussian")

    return img, gray, blur

def gmsd(ref, target):
    
    # first pass a 2x2 average filter
    
    ref = cv2.blur(ref,(2,2))
    target = cv2.blur(target,(2,2))
    
    # then downscale by a factor of two    
    
    ref = block_mean(ref,2)    
    target = block_mean(target,2)
    
    # reference gradient magnitude image from prewitt
    
    ref_conv_hx = convolve(ref, h_x)
    ref_conv_hy = convolve(ref, h_y)
    ref_grad_mag = np.sqrt( ( ref_conv_hx**2 ) + ( ref_conv_hy**2 ) )

    # target gradient magnitude image from prewitt

    dst_conv_hx = convolve(target, h_x)
    dst_conv_hy = convolve(target, h_y)
    dst_grad_mag = np.sqrt( ( dst_conv_hx**2 ) + ( dst_conv_hy**2 ) )
    
    #constant for numerical stability (see paper)
    
    c = 0.0026
    
    gms_map = ( 2 * ref_grad_mag * dst_grad_mag + c ) / ( ref_grad_mag**2 + dst_grad_mag**2 + c )
    
    gms_deviation = round(( np.sum(( gms_map-gms_map.mean() )**2 ) / gms_map.size )**0.5, 3 )

    print gms_deviation
    
    fig, axes = plt.subplots( nrows = 2, ncols = 3 )
    
    plt.subplot(231)
    plt.imshow(img)
    plt.title('Reference')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(232)
    plt.imshow(target, cmap = 'gray')
    plt.title('Distorted')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(233)
    plt.imshow(ref_grad_mag, cmap = 'gray')
    plt.title('Ref. Gradient Magnitude', size = 8)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(234)
    plt.imshow(dst_grad_mag, cmap = 'gray')
    plt.title('Dist. Gradient Magnitude', size = 8)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(235)
    plt.imshow(gms_map, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('GMS, GMSD: '+ str(gms_deviation))
    
    fig.delaxes(axes[-1,-1]) 
    
    plt.savefig('/home/cparr/Snow_Patterns/figures/gsmd/gsmd_house.png',
                bbox_inches = 'tight', dpi = 300, facecolor = 'skyblue')

# Prewitt kernels given in paper

h_x = [0.33, 0, -0.33,0.33, 0, -0.33, 0.33, 0, -0.33]
h_x = np.array(h_x).reshape(3,3)                  
h_y = np.flipud( np.rot90(h_x) )

inputs = input_data('/home/cparr/Downloads/jpeg2000_db/db/', 'paintedhouse.bmp',21)
img = inputs[0]
ref = inputs[1]
target = inputs[2]

gmsd(ref, target)

