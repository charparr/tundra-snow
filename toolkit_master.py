# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:26:13 2016

@author: cparr

This script uses a variety of image quality assessment (IQA) metrics to
determine the similiarity or lack thereof between two images.

"""

import glob
import re
import phasepack
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from itertools import combinations
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from scipy import signal
from scipy import fftpack
from matplotlib import six

def input_data(path_to_snow_data):
    
    """
    Return list of .tif files from the specified folder.
    Slicet them into ROIs and store in dictionaries.
    """
    
    file_list = glob.glob(str(path_to_snow_data) + '*.tif')
    
    for f in file_list:
        
        src = rasterio.open(f)
        snow_image = src.read(1)
        snow_image = np.ma.masked_values( snow_image, src.nodata )
        subset_256_avgfltr_2x2 = cv2.blur( snow_image[3200:3456,310:566],(2,2))
        
        polygons = subset_256_avgfltr_2x2[200::,100::]
        west_drift = subset_256_avgfltr_2x2[120:200]
        twin_drift = subset_256_avgfltr_2x2[40:120]
        no_drift = subset_256_avgfltr_2x2[0:40]
    
        name = f.split('/')[-1].rstrip('_snow_full.tif')
        name = name.strip('hv_')
        
        snow_dict[name] = subset_256_avgfltr_2x2
        polygon_dict[name] = polygons
        west_drift_dict[name] = west_drift
        twin_drift_dict[name] = twin_drift
        no_drift_dict[name] = no_drift        


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


def feature_sim(reference, target):
    
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
    # imgY = 0.299 * r + 0.587 * g + 0.114 * b

    # Constants provided by the authors
    t1 = 0.85
    t2 = 160
    
    # Phase congruency (PC) images. "PC...a dimensionless measure for the
    # significance of local structure.
    
    pc1 = phasepack.phasecong(reference, nscale = 4, norient = 4, 
                              minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                              
    pc2 = phasepack.phasecong(target, nscale = 4, norient = 4,
                              minWaveLength = 6, mult = 2, sigmaOnf=0.55)
                              
    pc1 = pc1[0]  # Reference PC map
    pc2 = pc2[0]  # Distorted PC map
    
    # Similarity of PC components
    s_PC = ( 2*pc1 + pc2 + t1 )  / ( pc1**2 + pc2**2 + t1 )
    
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    refgradX = cv2.Sobel(reference, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
    refgradY = cv2.Sobel(reference, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
    
    targradX = cv2.Sobel(target, cv2.CV_64F, dx = 1, dy = 0, ksize = -1)
    targradY = cv2.Sobel(target, cv2.CV_64F, dx = 0, dy = 1, ksize = -1)
    
    refgradient = np.maximum(refgradX, refgradY)    
    targradient = np.maximum(targradX, targradY)   
    
    #refgradient = np.sqrt(( refgradX**2 ) + ( refgradY**2 ))
    
    #targradient = np.sqrt(( targradX**2 ) + ( targradY**2 ))

    # The gradient magnitude similarity

    s_G = (2*refgradient + targradient + t2) / (refgradient**2 + targradient**2 + t2)
    
    s_L = s_PC * s_G  # luma similarity
    
    pcM = np.maximum(pc1,pc2)
        
    fsim = round( np.nansum( s_L * pcM) / np.nansum(pcM), 3)
    
    return fsim


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
        cw_ssim_map = cw_ssim_map.reshape(reference.shape[0],
                                          reference.shape[1])

        # Average the per pixel results
        cw_ssim_index = round( np.average(cw_ssim_map), 3)
        
        return cw_ssim_index, cw_ssim_map

 
def do_all_metrics(comparison_dict):
    
    pairs_lvl0 = [k for k in comparison_dict.iterkeys()]
    
    for p in pairs_lvl0:
        
        data_keys = [j for j in comparison_dict[p].iterkeys()]
        regex = re.compile('2')
        snow = [string for string in data_keys if re.match(regex, string)]
        
        comparison_dict[p]['MSE'] = round(mse(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]]),3)
                                        
        comparison_dict[p]['SSIM'] = round(ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]]),3)
                                        
        comparison_dict[p]['MSE Map'] = (comparison_dict[p][data_keys[0]] - 
                                          comparison_dict[p][data_keys[1]])**2
                                          
        comparison_dict[p]['SSIM Map'] = ssim(comparison_dict[p][data_keys[0]], 
                                          comparison_dict[p][data_keys[1]],
                                            full = True)[1]
    
        comparison_dict[p]['CW-SSIM'] = cw_ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]], 40)[0]
                                        
        comparison_dict[p]['CW-SSIM Map'] = cw_ssim(comparison_dict[p][data_keys[0]],
                                        comparison_dict[p][data_keys[1]], 40)[1]
                                        
        comparison_dict[p]['GMS'] = gmsd(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[0]
                                        
        comparison_dict[p]['GMS Map'] = gmsd(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[1]

    
        comparison_dict[p][snow[0]+' DCT Map'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[0]
                                        
        comparison_dict[p][snow[1]+' DCT Map'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[1]
                                        
        comparison_dict[p][snow[0]+' DCT Curve'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[2]
                                        
        comparison_dict[p][snow[1]+' DCT Curve'] = discrete_cosine(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])[3]
                                        
                                        
        comparison_dict[p]['FSIM'] = feature_sim(comparison_dict[p][snow[0]],
                                        comparison_dict[p][snow[1]])
        
def make_plots(roi, comparison_dict):
    for j in comparison_dict.keys():
      
        titles = [x for x in comparison_dict[j] if
        type(comparison_dict[j][x]) != float and
        len(comparison_dict[j][x].shape) == 2]
        titles = sorted(titles)
        titles.insert(1, titles.pop(2))
       
        curves = [x for x in comparison_dict[j] if
        type(comparison_dict[j][x]) != float and
        len(comparison_dict[j][x].shape) == 1]
       
        i=1
       
        fig, axes = plt.subplots(nrows = 3, ncols = 3)
        
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
       
        plt.suptitle(roi + ' ' + j + ' Snow Pattern Comparison')
       
        for name in titles:
            ax1 = plt.subplot(3,3,i)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.set_title(name, fontsize = 9)
            im1 = ax1.imshow(comparison_dict[j][name], cmap = 'viridis')
            i+=1
           
        plt.subplot(3,3,i)
        plt.plot(comparison_dict[j][curves[0]], label = curves[0][0:4])
        plt.plot(comparison_dict[j][curves[1]], label = curves[1][0:4])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, fontsize = 7,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.yscale('log')
        plt.xlim([0, len(comparison_dict[j][curves[0]])])
        #plt.ylim([0,2])
        plt.xticks([0,len(comparison_dict[j][curves[0]]) / 2,
                    len(comparison_dict[j][curves[0]])], size = 6)
        #plt.yticks([])

       
        textstr = 'MSE = %s, SSIM = %s, CW-SSIM = %s, GMS = %s, FSIM = %s' % (
        comparison_dict[j]['MSE'],comparison_dict[j]['SSIM'],
        comparison_dict[j]['CW-SSIM'],comparison_dict[j]['GMS'],
        comparison_dict[j]['FSIM'])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
       
        fig.text(0.15, 0.05, textstr, fontsize=8,
               verticalalignment='top', bbox=props)
     
        plt.savefig('/home/cparr/Snow_Patterns/figures/hv_toolbox_results/' +
                    j + '_' + roi + '.png',
                    dpi = 300, bbox_inches = 'tight')                                        

def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=12,
                     header_color='#236192', row_colors=['#C7C9C7', 'w'],
                     edge_color='w',bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText = data.values, rowLabels = data.index,
                         bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='#FFCD00')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

       
def create_pairs(dict_w_snow_data, comparison_dict):

    for pair in combinations(dict_w_snow_data.iteritems(), 2):
        comparison_dict[pair[0][0] + " vs. " + pair[1][0]] = {}
        comparison_dict[pair[0][0] + " vs. " + pair[1][0]][pair[0][0]] = pair[0][1]
        comparison_dict[pair[0][0] + " vs. " + pair[1][0]][pair[1][0]] = pair[1][1]
        del pair
        

# Globals go here                                        
snow_dict = defaultdict(dict)
polygon_dict = defaultdict(dict)
west_drift_dict = defaultdict(dict)
twin_drift_dict = defaultdict(dict)
no_drift_dict = defaultdict(dict)

snow_comparison_dict = defaultdict(dict)
polygon_comparison_dict = defaultdict(dict)
west_drift_comparison_dict = defaultdict(dict)
twin_drift_comparison_dict = defaultdict(dict)
no_drift_comparison_dict = defaultdict(dict)

data_dicts = [snow_dict, polygon_dict, west_drift_dict, 
             twin_drift_dict, no_drift_dict]
             
comp_dicts = [snow_comparison_dict, polygon_comparison_dict,
              west_drift_comparison_dict,twin_drift_comparison_dict,
              no_drift_comparison_dict]
             
all_names = ['snow','polygon','west_drift','twin_drift','no_drift']

snow_df = pd.DataFrame()
polygon_df = pd.DataFrame()
west_df = pd.DataFrame()
twin_df = pd.DataFrame()
no_drift_df = pd.DataFrame()

dfs = [snow_df, polygon_df, west_df, twin_df, no_drift_df]

# this builds the actual comparison

input_data('/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_on/full_extent/')

for d in zip(data_dicts, comp_dicts, all_names):
    
    create_pairs(d[0],d[1])

    do_all_metrics(d[1])

    # provide a string to choose figure filename
    make_plots(d[2],d[1])
    
    del d

#==============================================================================
# for d in zip(dfs, comp_dicts, all_names):
#     
#     d[0] = pd.DataFrame.from_dict(d[1])
#     d[0] = d[0].T
#     d[0].index.name = 'Comparison Years'
#     d[0]['Region'] = 'd[2]'
#     d[0].set_index('Region', append=True, inplace=True)
#     d[0] = d[0].reorder_levels(['Region','Comparison Years'])
#     for c in d[0].columns:
#         if type(d[0][c][0]) != float or d[0][c].isnull().any()==True:
#             del d[0][c]
# 
#==============================================================================
#polygon_df = pd.DataFrame.from_dict(polygon_comparison_dict)
#polygon_df = polygon_df.T
#polygon_df.index.name = 'Comparison Years'
#polygon_df['Region'] = 'Polygon'
#polygon_df.set_index('Region', append=True, inplace=True)
#polygon_df = polygon_df.reorder_levels(['Region','Comparison Years'])
#for c in polygon_df.columns:
#    if type(polygon_df[c][0]) != float or polygon_df[c].isnull().any()==True:
#        del polygon_df[c]
#west_drift_df = pd.DataFrame.from_dict(west_drift_comparison_dict)
#west_drift_df = west_drift_df.T
#west_drift_df.index.name = 'Comparison Years'
#west_drift_df['Region'] = 'West Drift'
#west_drift_df.set_index('Region', append=True, inplace=True)
#west_drift_df = west_drift_df.reorder_levels(['Region','Comparison Years'])
#for c in west_drift_df.columns:
#    if type(west_drift_df[c][0]) != float or west_drift_df[c].isnull().any()==True:
#        del west_drift_df[c]
#twin_drift_df = pd.DataFrame.from_dict(twin_drift_comparison_dict)
#twin_drift_df = twin_drift_df.T
#twin_drift_df.index.name = 'Comparison Years'
#twin_drift_df['Region'] = 'Twin Drifts'
#twin_drift_df.set_index('Region', append=True, inplace=True)
#twin_drift_df = twin_drift_df.reorder_levels(['Region','Comparison Years'])
#for c in twin_drift_df.columns:
#    if type(twin_drift_df[c][0]) != float or twin_drift_df[c].isnull().any()==True:
#        del twin_drift_df[c]
#no_drift_df = pd.DataFrame.from_dict(no_drift_comparison_dict)
#no_drift_df = no_drift_df.T
#no_drift_df.index.name = 'Comparison Years'
#no_drift_df['Region'] = 'No Drifts'
#no_drift_df.set_index('Region', append=True, inplace=True)
#no_drift_df = no_drift_df.reorder_levels(['Region','Comparison Years'])
#for c in no_drift_df.columns:
#    if type(no_drift_df[c][0]) != float or no_drift_df[c].isnull().any()==True:
#        del no_drift_df[c]

#master_df = no_drift_df.append(polygon_df)
#master_df = master_df.append(twin_drift_df)
#master_df = master_df.append(west_drift_df)
#master_df = master_df.append(snow_df)
#
#del polygon_df
#del twin_drift_df
#del west_drift_df
#del snow_df

#render_mpl_table(master_df)
#plt.savefig('/home/cparr/testtable.png')