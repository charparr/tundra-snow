
# coding: utf-8

# In[1]:

# Data wrangling libraries
import pandas as pd
import numpy as np
from collections import defaultdict

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import six
get_ipython().magic(u'matplotlib qt')

# View a DataFrame in browser
import webbrowser
from tempfile import NamedTemporaryFile

# Analysis Libraries
import scipy
import cv2
from scipy import signal
from scipy.spatial import *
from scipy.ndimage import *
from skimage.transform import *
from skimage.morphology import *
from skimage.util import *
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.transform import AffineTransform
from skimage.transform import warp


# In[2]:

### Similarity Test Functions ###

def procrustes_analysis(data):
    for d in data:
        mtx1, mtx2, disparity = procrustes(data[0], d)
        # disparity is the sum of the square errors
        # mtx2 is the optimal matrix transformation
        disp_vals.append(disparity.round(3))

def make_quadrants(data):
    
    q = data[0].shape[0] / 2
    for d in data:
        tl, tr, ll, lr = d[:q, :q], d[q:, :q], d[:q, q:], d[q:, q:]
        top_lefts.append(tl)
        top_rights.append(tr)
        low_lefts.append(ll)
        low_rights.append(tr)
        
def structural_sim(data):
    
    for d in data:
        ssim_vals.append( ssim ( data[0], d ).round( 2 ))
        
        ssim_maps.append( ssim ( data[0], d, full  = True )[1] )
        
def reg_mse(data):
    for d in data:
        mse_vals.append(( mse ( data[0], d )).round(2))
        mse_maps.append((data[0] - d) ** 2)

        
def imse( data ):
    
    for d in data:

        unique_vals_and_counts = np.round( np.unique( data[0], return_counts = True ), 1 )
        vals = np.array( unique_vals_and_counts[0], dtype = 'float32' )
        counts = np.array( unique_vals_and_counts[1], dtype = 'float32' )
        num_pixels = data[0].size

        shannons = np.round( np.divide( counts, num_pixels ), 6 )
        info_vals = np.round( np.log(1/shannons), 2)

        unique_info_vals = zip(vals,info_vals)
        trans_dct = {}

        for v in unique_info_vals:
            trans_dct[v[0]] = v[1]

        infomap = np.copy( data[0] )
        for k, v in trans_dct.iteritems(): infomap[data[0] == k] = v

        imse_map = (( infomap * data[0] ) - ( infomap * d )) ** 2
        imse_maps.append(imse_map)
        
        err = np.sum( imse_map )
        err /= float(data[0].shape[0] * data[0].shape[1])

        imse_vals.append( np.round(err, 2 ))

# Complex Wavelet SSIM

def cw_ssim_value(data, width):
        """Compute the complex wavelet SSIM (CW-SSIM) value from the reference
        image to the target image.
        Args:
          target (str or PIL.Image): Input image to compare the reference image
          width: width for the wavelet convolution (default: 30)
        Returns:
          Computed CW-SSIM float value.
        """

        # Define a width for the wavelet convolution
        widths = np.arange(1, width+1)

        for d in data:
        
            # Use the image data as arrays
            sig1 = np.asarray(data[0].ravel())
            sig2 = np.asarray(d.ravel())

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
            ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
            ssim_map = ssim_map.reshape(32,32)
            cw_ssim_maps.append(ssim_map)

            # Average the per pixel results
            index = round( np.average(ssim_map), 2) 
            cw_ssim_vals.append(index)
        
def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')

# Function to display DataFrame in new browser tab.

def df_window(df):
    
    with NamedTemporaryFile(delete=False, suffix='.html') as f:
        df.to_html(f)
    webbrowser.open(f.name)
    
def plot_continuous(names, data):

    fig, axes = plt.subplots( nrows = 3, ncols = 5 )
    fig.suptitle( 'Fidelity Tests of Continuous Patterns' )
    
    for p, dat, ax in zip( names, data, axes.flat ):
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p,fontsize = 6)
    
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ([-1,1]) )
    cax.tick_params(labelsize = 6)
    
def plot_tests(names, test_vals, test_name, data, rows, cols, cmin, cmax):
    
    fig, axes = plt.subplots( nrows = 3, ncols = 5 )
    for p, v, dat, ax in zip( names, test_vals, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = cmin, vmax = cmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p + " " + test_name + str(v), fontsize = 6)

    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [cmin, cmax] ) )
    cax.tick_params(labelsize = 6)

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#236192', row_colors=['#C7C9C7', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText = data.values, bbox=bbox, colLabels=data.columns, **kwargs)

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


# In[3]:

# Reference Patterns

pi_cycles = np.linspace( -np.pi, np.pi, 512 )
pi_cycles = np.append( pi_cycles, pi_cycles )
pi_cycles = pi_cycles.reshape( 32, 32 )

sine = np.sin( pi_cycles )
cosine = np.cos( pi_cycles )


# In[4]:

'''
Warping a base pattern of continuous data.
Applying MSE, SSIM, and Procrustes.
Zipping all of these values into a mega - list.
Afterward, create a dict and a DataFrame by iterating through mega - list.
'''

ctswarp_data = []

# Initialize lists for metrics.

mse_vals = []
ssim_vals = []
disp_vals = []
top_lefts = []
top_rights = []
low_lefts = []
low_rights = []
imse_vals = []
imse_maps = []
mse_maps = []
ssim_maps = []
mse_maps = []
cw_ssim_vals = []
cw_ssim_maps = []

# Create the test patterns

def warp_continuous(pattern):
    
    ctswarp_data.append(pattern)
    rows, cols = pattern.shape
    mu = pattern.mean()
    sigma = pattern.std()

    # 90 degree rotation
    rotate90 = np.rot90(pattern)
    ctswarp_data.append(rotate90)
    
    #45 degree rotation
    oblique = rotate(pattern, 45)
    b = oblique == 0
    oblique[b] = np.random.normal(mu, sigma, size=b.sum())
    ctswarp_data.append(oblique)
    
    # morphological dilation and erosion
    morph_dilation = dilation(pattern)
    morph_erosion = erosion(pattern)
    ctswarp_data.append(morph_dilation)
    ctswarp_data.append(morph_erosion)
    
    # flip up and down, basically a phase shift
    inverse = np.flipud(pattern)
    ctswarp_data.append(inverse)
    
    # a shift or translation
    shift_M = np.float32([[1,0,1],[0,1,0]])
    shifted = cv2.warpAffine(pattern,shift_M,(cols,rows))
    ctswarp_data.append(shifted)
    
    # randomly shuffle rows of array, create a random frequency
    permutation = np.random.permutation(pattern)
    ctswarp_data.append(permutation)
    
    # sine warp
    # basically sine of sine...reduces the intensity 
    sine_of  = np.sin(pattern)
    ctswarp_data.append(sine_of)
    
    # cosine
    ctswarp_data.append(cosine)
    
    # Random between -1 and 1
    random_abs1 = np.random.uniform(-1, 1, [32,32])
    ctswarp_data.append(random_abs1)
    
    # Gaussian noise
    mu = 0
    sigma = 0.32
    gauss_abs1 = np.random.normal(mu, sigma, (32,32))
    ctswarp_data.append(gauss_abs1)
    
    # Random Affine Transformation
    c = np.random.random_sample(( 6, ))
    m = np.append( c, ( 0,0,1 ) )
    m = m.reshape( 3,3 )
    aff_t = AffineTransform( matrix = m )
    random_aff_warp = warp( pattern, aff_t )
    b = random_aff_warp == 0
    random_aff_warp[b] = np.random.normal(mu, sigma, size=b.sum())
    ctswarp_data.append( random_aff_warp )
    
    # Additive Gaussian Noise
    noise = random_noise( pattern, mode = 'gaussian' )
    ctswarp_data.append( noise )
    
    # More Additive Gaussian Noise
    more_noise = random_noise(random_noise(random_noise(random_noise( noise, mode = 'gaussian' ))))
    ctswarp_data.append( more_noise )
    
# Plot titles and dict keys
ctswarp_names = ['Reference', 'Rotate 90', 'Rotate 45', 'Dilation',
                 'Erosion', 'Y Reflection', 'X Shift', 'Row Shuffle',
                 'Sin()', 'Cosine', 'Rand.', 'Gauss',
                 'Rand. Affine', 'Gauss Noise','More Noise']

warp_continuous( sine )

# Call Metrics on list of test patterns

structural_sim( ctswarp_data )
reg_mse( ctswarp_data )
procrustes_analysis( ctswarp_data )
make_quadrants( ctswarp_data )
imse(ctswarp_data)
cw_ssim_value(ctswarp_data, 30)

# Zip names, data, metrics, quadrants into a mega list!
# Generally this is indavisable because it relies on index locations...in the next cell we will make a dictionary.
cts_zip = zip(ctswarp_names, ctswarp_data, mse_vals, ssim_vals, disp_vals, top_lefts, top_rights, low_lefts, low_rights, 
               imse_vals, imse_maps, mse_maps, ssim_maps, cw_ssim_vals, cw_ssim_maps)


# In[5]:

continuous_dict = defaultdict(dict)

# Making a look up dictionary from all the patterns and their comparison scores.
# zipped list [i][0] is namne, 1 is full array, 2 is mse val, 3 is SSIM, 4 is PD,
# 5 through 8 are quadrants
# 9 is IMSE, 10 is IMSE Map


def to_dict_w_hists( data_dict, keys, data_zip ):

    i = 0
    while i < len(keys):

        data_dict[keys[i]]['name'] = data_zip[i][0]

        data_dict[keys[i]]['arrays'] = {}

        data_dict[keys[i]]['arrays']['full'] = {}
        data_dict[keys[i]]['arrays']['full']['array'] = data_zip[i][1]
        data_dict[keys[i]]['arrays']['full']['numpy hist'] = np.histogram( data_zip[i][1] )
        data_dict[keys[i]]['arrays']['full']['cv2 hist'] = np_hist_to_cv( np.histogram( data_zip[i][1] ) )
        
        data_dict[keys[i]]['MSE'] = round(data_zip[i][2], 2)
        data_dict[keys[i]]['SSIM'] = round(data_zip[i][3], 2)
        data_dict[keys[i]]['Procrustres Disparity'] = round(data_zip[i][4], 2)

        data_dict[keys[i]]['arrays']['top left'] = {}
        data_dict[keys[i]]['arrays']['top left']['array'] = data_zip[i][5]
        data_dict[keys[i]]['arrays']['top left']['numpy hist'] = np.histogram( data_zip[i][5] )
        data_dict[keys[i]]['arrays']['top left']['cv2 hist'] = np_hist_to_cv( np.histogram( data_zip[i][5] ) )

        data_dict[keys[i]]['arrays']['top right'] = {}
        data_dict[keys[i]]['arrays']['top right']['array'] = data_zip[i][6]
        data_dict[keys[i]]['arrays']['top right']['numpy hist'] = np.histogram( data_zip[i][6] )
        data_dict[keys[i]]['arrays']['top right']['cv2 hist'] = np_hist_to_cv( np.histogram( data_zip[i][6] ) )

        data_dict[keys[i]]['arrays']['low left'] = {}
        data_dict[keys[i]]['arrays']['low left']['array'] = data_zip[i][7]
        data_dict[keys[i]]['arrays']['low left']['numpy hist'] = np.histogram( data_zip[i][7] )
        data_dict[keys[i]]['arrays']['low left']['cv2 hist'] = np_hist_to_cv( np.histogram( data_zip[i][7] ) )

        data_dict[keys[i]]['arrays']['low right'] = {}
        data_dict[keys[i]]['arrays']['low right']['array'] = data_zip[i][8]
        data_dict[keys[i]]['arrays']['low right']['numpy hist'] = np.histogram( data_zip[i][8] )
        data_dict[keys[i]]['arrays']['low right']['cv2 hist'] = np_hist_to_cv( np.histogram( data_zip[i][8] ) )    

        data_dict[keys[i]]['IMSE'] = round(data_zip[i][9], 2)
        data_dict[keys[i]]['IMSE Map'] = data_zip[i][10]
        data_dict[keys[i]]['MSE Map'] = data_zip[i][11]
        data_dict[keys[i]]['SSIM Map'] = data_zip[i][12]
        data_dict[keys[i]]['CW SSIM'] = data_zip[i][13]
        data_dict[keys[i]]['CW SSIM Map'] = data_zip[i][14]

        # Histogram Comparisons

        # Bhattacharyya
        data_dict[keys[i]]['Bhattacharyya Full'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['full']['cv2 hist'],
                data_dict[keys[0]]['arrays']['full']['cv2 hist'],
                cv2.cv.CV_COMP_BHATTACHARYYA), 2)

        data_dict[keys[i]]['Bhattacharyya UL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top left']['cv2 hist'],
                cv2.cv.CV_COMP_BHATTACHARYYA), 2)

        data_dict[keys[i]]['Bhattacharyya UR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top right']['cv2 hist'],
                cv2.cv.CV_COMP_BHATTACHARYYA), 2)

        data_dict[keys[i]]['Bhattacharyya LL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low left']['cv2 hist'],
                cv2.cv.CV_COMP_BHATTACHARYYA), 2)   

        data_dict[keys[i]]['Bhattacharyya LR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low right']['cv2 hist'],
                cv2.cv.CV_COMP_BHATTACHARYYA), 2)

        # Chi Square
        data_dict[keys[i]]['Chi Square Full'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['full']['cv2 hist'],
                data_dict[keys[0]]['arrays']['full']['cv2 hist'],
                cv2.cv.CV_COMP_CHISQR), 2)

        data_dict[keys[i]]['Chi Square UL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top left']['cv2 hist'],
                cv2.cv.CV_COMP_CHISQR), 2)

        data_dict[keys[i]]['Chi Square UR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top right']['cv2 hist'],
                cv2.cv.CV_COMP_CHISQR), 2)

        data_dict[keys[i]]['Chi Square LL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low left']['cv2 hist'],
                cv2.cv.CV_COMP_CHISQR), 2)

        data_dict[keys[i]]['Chi Square LR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low right']['cv2 hist'],
                cv2.cv.CV_COMP_CHISQR), 2)

        # Correlation
        data_dict[keys[i]]['Correlation Full'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['full']['cv2 hist'],
                data_dict[keys[0]]['arrays']['full']['cv2 hist'],
                cv2.cv.CV_COMP_CORREL), 2)

        data_dict[keys[i]]['Correlation UL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top left']['cv2 hist'],
                cv2.cv.CV_COMP_CORREL), 2)

        data_dict[keys[i]]['Correlation UR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['top right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['top right']['cv2 hist'],
                cv2.cv.CV_COMP_CORREL), 2)

        data_dict[keys[i]]['Correlation LL'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low left']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low left']['cv2 hist'],
                cv2.cv.CV_COMP_CORREL), 2)

        data_dict[keys[i]]['Correlation LR'] = round(cv2.compareHist(
                data_dict[keys[i]]['arrays']['low right']['cv2 hist'],
                data_dict[keys[0]]['arrays']['low right']['cv2 hist'],
                cv2.cv.CV_COMP_CORREL), 2)

        i = i + 1

to_dict_w_hists( continuous_dict, ctswarp_names, cts_zip )

cts_df = pd.DataFrame.from_dict(continuous_dict)
cts_df = cts_df.transpose()


# In[6]:

# Histogram Scores

hist_scores = cts_df.loc[:,['name', 'Bhattacharyya UL','Bhattacharyya UR','Bhattacharyya LL',
'Bhattacharyya LR', 'Bhattacharyya Full','Correlation UL','Correlation UR','Correlation LL',
'Correlation LR', 'Correlation Full','Chi Square UL','Chi Square UR','Chi Square LL',
'Chi Square LR', 'Chi Square Full']]

hist_scores['Mean Bhattacharyya'] = np.round(hist_scores[['Bhattacharyya UL','Bhattacharyya UR',
                                            'Bhattacharyya LL', 'Bhattacharyya LR']].mean(axis = 1),2)

hist_scores['Mean Correlation'] = np.round(hist_scores[['Correlation UL','Correlation UR',
                                            'Correlation LL', 'Correlation LR']].mean(axis = 1),2)

hist_scores['Mean Chi Square'] = np.round(hist_scores[['Chi Square UL','Chi Square UR',
                                            'Chi Square LL', 'Chi Square LR']].mean(axis = 1),2)

hist_scores = hist_scores[['Mean Bhattacharyya', 'Mean Chi Square','Mean Correlation']]


hist_scores = hist_scores.sort_values('Mean Bhattacharyya')

#df_window(hist_scores)


# In[7]:

# Continuous Scores and Ranks

cts_scores = cts_df.copy()
cts_scores['Pattern'] = cts_df['name']
cts_scores = cts_scores[['Pattern', 'MSE', 'SSIM', 'IMSE', 'Procrustres Disparity', 'CW SSIM']]
cts_scores = cts_scores.sort_values( 'CW SSIM', ascending = False )

ranks = cts_scores.copy()
ranks['Pattern'] = cts_df['name']
ranks['MSE Rank'] = np.round(cts_scores['MSE'].rank(ascending=True))
ranks['SSIM Rank'] = cts_scores['SSIM'].rank(ascending=False)
ranks['IMSE Rank'] = np.round(cts_scores['IMSE'].rank(ascending=True))
ranks['CW-SSIM Rank'] = cts_scores['CW SSIM'].rank(ascending=False)
ranks['Disparity Rank'] = cts_scores['Procrustres Disparity'].rank()
ranks['Bhattacharyya Rank'] = hist_scores['Mean Bhattacharyya'].rank(ascending=True)
ranks['Chi Square Rank'] = hist_scores['Mean Chi Square'].rank(ascending=True)
ranks['Correlation Rank'] = hist_scores['Mean Correlation'].rank(ascending=False)
del ranks['MSE']
del ranks['IMSE']
del ranks['SSIM']
del ranks['CW SSIM']
del ranks ['Procrustres Disparity']
ranks = ranks.sort_values('CW-SSIM Rank')

#df_window(cts_scores)


# In[10]:

render_mpl_table(ranks)
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_ranks.png', bbox_inches = 'tight', dpi = 300)
plt.close()

render_mpl_table(cts_scores)
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_scores.png', bbox_inches = 'tight', dpi = 300)
plt.close()

plot_continuous( ctswarp_names, ctswarp_data )
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_test_patterns.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()

# names, test_vals, test_name, data, rows, cols, cmin, cmax

plot_tests( ctswarp_names, imse_vals, " IMSE: ", imse_maps, 4, 4, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_imse_map.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()

plot_tests( ctswarp_names, mse_vals, " MSE: ", mse_maps, 4, 4, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_mse_map.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()

plot_tests( ctswarp_names, ssim_vals, " SSIM: ", ssim_maps, 4, 4, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_ssim_map.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()

plot_tests( ctswarp_names, cw_ssim_vals, " CW SSIM: ", cw_ssim_maps, 4, 4, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/continuous_test/continuous_cw_ssim_map.png', bbox_inches = 'tight',
            dpi = 300)

