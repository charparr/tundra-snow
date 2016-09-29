# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:26:13 2016

@author: cparr

This script uses a variety of image quality assessment (IQA) metrics to
determine the similiarity or lack thereof between two images.

"""

import glob
import re
from collections import defaultdict
import numpy as np
from itertools import combinations
#from scipy import ndimage
#import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.filters import scharr
import phasepack
import rasterio
from scipy import signal
from scipy import fftpack
#from scipy.spatial import *
#from scipy.ndimage import *
#from skimage.transform import *
#from skimage.morphology import *
#from skimage.util import *

def input_data(path_to_snow_data):
    
    """
    Return list of .tif files from the specified folder.
    """
    
    file_list = glob.glob(str(path_to_snow_data) + '*.tif')
    
    for f in file_list:
        
        src = rasterio.open(f)
        snow_image = src.read(1)
        snow_image = np.ma.masked_values( snow_image, src.nodata )
        subset_256_avgfltr_2x2 = cv2.blur( snow_image[3200:3456,310:566],(2,2))
    
        name = f.split('/')[-1].rstrip('_snow_full.tif')
        
        snow_dict[name] = subset_256_avgfltr_2x2

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
        
def discrete_cosine(reference, target):
    
    reference_transform = fftpack.dct( reference )
    target_transform = fftpack.dct( target )
    reference_curve = reference_transform.mean(axis = 1)
    target_curve = target_transform.mean(axis = 1)

    return reference_transform, target_transform, reference_curve, target_curve


def gmsd(reference, target):
    
    """
    Return a map of Gradient Magnitude Similarity and a global index measure.
    
    Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). 
    Gradient magnitude similarity deviation: A highly efficient perceptual
    image quality index. IEEE Transactions on Image Processing, 23(2), 668–695.
    http://doi.org/10.1109/TIP.2013.2293423
    """
        
    # Prewitt kernels given in paper
    h_x = [0.33, 0, -0.33,0.33, 0, -0.33, 0.33, 0, -0.33]
    h_x = np.array(h_x).reshape(3,3)                  
    h_y = np.flipud( np.rot90(h_x) )
    
    # Create gradient magnitude images with Prewitt kernels
    ref_conv_hx = convolve(reference, h_x)
    ref_conv_hy = convolve(reference, h_y)
    ref_grad_mag = np.sqrt( ( ref_conv_hx**2 ) + ( ref_conv_hy**2 ) )
    
    dst_conv_hx = convolve(target, h_x)
    dst_conv_hy = convolve(target, h_y)
    dst_grad_mag = np.sqrt( ( dst_conv_hx**2 ) + ( dst_conv_hy**2 ) )
        
    c = 0.0026  # Constant provided by the authors
    
    gms_map = ( 2 * ref_grad_mag * dst_grad_mag + c ) / ( ref_grad_mag**2 + dst_grad_mag**2 + c )
    
    gms_index = round(( np.sum(( gms_map-gms_map.mean() )**2 ) / gms_map.size )**0.5, 3 )    
    
    return gms_index, gms_map


def feature_sim(reference_img, dist_img):
    
    """
    Return the Feature Similarity Index (FSIM).
    Can also return FSIMc for color images
    
    Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). 
    FSIM: A feature similarity index for image quality assessment. 
    IEEE Transactions on Image Processing, 20(8), 2378–2386. 
    http://doi.org/10.1109/TIP.2011.2109730
    """
    
    # Convert the input images to YIQ color space
    # Y is the luma compenent, i.e. B & W
    r, g, b = reference_img[:,:,0], reference_img[:,:,1], reference_img[:,:,2]
    imgY = 0.299 * r + 0.587 * g + 0.114 * b
    imgI = 0.596 * r - 0.275 * g - 0.321 * b
    imgQ = 0.212 * r - 0.523 * g + 0.311 * b
    
    r_d, g_d, b_d = dist_img[:,:,0], dist_img[:,:,1], dist_img[:,:,2]
    dstY = 0.299 * r_d + 0.587 * g_d + 0.114 * b_d
    dstI = 0.596 * r_d - 0.275 * g_d - 0.321 * b_d
    dstQ = 0.212 * r_d - 0.523 * g_d + 0.311 * b_d
    
    # Constants provided by the authors
    t1 = 0.85
    t2 = 160
    t3 = 200
    t4 = 200
    
    # Similarity of the chromatic features (I and Q channels)
    s_Q = ( 2*imgQ + dstQ + t4 )  / ( imgQ**2 + dstQ**2 + t4 )
    s_I = ( 2*imgI + dstI + t3 )  / ( imgI**2 + dstI**2 + t3 )
    
    # Phase congruency (PC) images. "PC...a dimensionless measure for the
    # significance of local structure.
    
    pc1 = phasepack.phasecong(imgY, nscale = 4, norient = 4, minWaveLength = 6,
                              mult = 2, sigmaOnf=0.55)
    pc2 = phasepack.phasecong(dstY, nscale = 4, norient = 4, minWaveLength = 6,
                              mult = 2, sigmaOnf=0.55)
    pc1 = pc1[0]  # Reference PC map
    pc2 = pc2[0]  # Distorted PC map
    
    # Similarity of PC components
    s_PC = ( 2*pc1 + pc2 + t1 )  / ( pc1**2 + pc2**2 + t1 )
    
    # Compute gradient magnitude maps using the Scharr operator
    g1 = scharr( imgY )
    g2 = scharr( dstY )
    
    # The gradient magnitude similarity
    s_G = ( 2*g1 + g2 + t2 )  / ( g1**2 + g2**2 + t2 )
    
    s_L = s_PC * s_G  # Luminance similarity

    s_C = s_I * s_Q  # Chromatic simlarity
    
    pcM = np.maximum(pc1,pc2)  # Maximum PC is used to weight s_L...basically
    #  PC structures are significant
        
    fsim = round( np.nansum( s_L * pcM) / np.nansum(pcM), 3)
    
    fsimc = round( np.nansum( s_L * s_C**0.3 * pcM) / np.nansum(pcM), 3)
#  

def cw_ssim(reference, target, width):
    
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.
        Args:
          reference and target images
          width: width for the wavelet convolution (default: 30)
        Returns:
          Computed CW-SSIM float value and map.
        """

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)
        
        # Use the image data as arrays
        sig1 = np.ravel(reference)
        sig2 = np.ravel(target)

        # Convolution
        cwtmatr1 = signal.cwt(sig1, signal.ricker, widths)
        cwtmatr2 = signal.cwt(sig2, signal.ricker, widths)

        # Compute the first term
        c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
        c1_2 = np.square(abs(cwtmatr1))
        c2_2 = np.square(abs(cwtmatr2))
        num_ssim_1 = 2 * np.sum(c1c2, axis=0) + 0.01
        den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + 0.01

        # Compute the second term
        c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
        num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + 0.01
        den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + 0.01

        # Construct the result
        cw_ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
        cw_ssim_map = cw_ssim_map.reshape(reference.shape[0],reference.shape[1])

        # Average the per pixel results
        cw_ssim_index = round( np.average(cw_ssim_map), 3)
        
        return cw_ssim_index, cw_ssim_map

def mse_and_ssim():
    
    pairs_lvl0 = [k for k in comparison_dict.iterkeys()]
    
    for p in pairs_lvl0:
        
        data_keys = [j for j in comparison_dict[p].iterkeys()]
        
        comparison_dict[p]['MSE'] = round(mse(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]]),3)
                                        
        comparison_dict[p]['SSIM'] = round(ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]]),3)
                                        
        comparison_dict[p]['MSE Map'] = (comparison_dict[p][data_keys[0]] - 
                                          comparison_dict[p][data_keys[1]])**2
                                          
        comparison_dict[p]['SSIM Map'] = ssim(comparison_dict[p][data_keys[0]], 
                                          comparison_dict[p][data_keys[1]],
                                            full = True)[1]
                                            
def do_cw_ssim():
    
    pairs_lvl0 = [k for k in comparison_dict.iterkeys()]
    
    for p in pairs_lvl0:
        
        data_keys = [j for j in comparison_dict[p].iterkeys()]
        
        comparison_dict[p]['CW-SSIM'] = cw_ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]], 20)[0]
                                        
        comparison_dict[p]['CW-SSIM Map'] = cw_ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]], 20)[1]
                                        
def do_gmsd():
    
    pairs_lvl0 = [k for k in comparison_dict.iterkeys()]
    
    for p in pairs_lvl0:
        
        data_keys = [j for j in comparison_dict[p].iterkeys()]
        regex = re.compile('h')
        snow = [string for string in data_keys if re.match(regex, string)]
        
        comparison_dict[p]['GMS'] = gmsd(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[0]
                                        
        comparison_dict[p]['GMS Map'] = gmsd(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[1]
                                        
def do_dct():
    
    pairs_lvl0 = [k for k in comparison_dict.iterkeys()]
    
    for p in pairs_lvl0:
        
        data_keys = [j for j in comparison_dict[p].iterkeys()]
        regex = re.compile('h')
        snow = [string for string in data_keys if re.match(regex, string)]
        
        comparison_dict[p][snow[0]+' DCT Map'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[0]
                                        
        comparison_dict[p][snow[1]+' DCT Map'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[1]
                                        
        comparison_dict[p][snow[0]+' DCT Curve'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[2]
                                        
        comparison_dict[p][snow[1]+' DCT Curve'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[3]
                                            
    
snow_dict = defaultdict(dict)

input_data('/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_on/full_extent/')

comparison_dict = defaultdict(dict)

for pair in combinations(snow_dict.iteritems(), 2):
    comparison_dict[pair[0][0] + "  vs.  " + pair[1][0]] = {}
    comparison_dict[pair[0][0] + "  vs.  " + pair[1][0]][pair[0][0]] = pair[0][1]
    comparison_dict[pair[0][0] + "  vs.  " + pair[1][0]][pair[1][0]] = pair[1][1]
    del pair

mse_and_ssim()
do_cw_ssim()
do_gmsd()
do_dct()





