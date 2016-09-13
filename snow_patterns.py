
# coding: utf-8

# In[1]:

# Data wrangling libraries
import pandas as pd
import numpy as np
import rasterio
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
from scipy import fftpack
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
          to. This may be a PIL Image object or, to save time, an SSIMImage
          object (e.g. the img member of another SSIM object).
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
            ssim_map = ssim_map.reshape(512,512)
            cw_ssim_maps.append(ssim_map)

            # Average the per pixel results
            index = round( np.average(ssim_map), 2) 
            cw_ssim_vals.append(index)
            
# Mag. Spectrum

def transform( data ):
    
    for d in data:
        f = np.fft.fft2( d )
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        mag_maps.append( magnitude_spectrum )
        
        mean_freq = magnitude_spectrum.mean(axis = 0)
        mean_freq = np.absolute(mean_freq)
        freqs.append(mean_freq)
        
def disccost( data ):
    
    for d in data:
        y = fftpack.dct( d )
        dct_maps.append( y )
        yc = y.mean(axis = 1 )
        dct_curves.append( yc )
        
def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')

### Plotting Functions ###

# Function to display DataFrame in new browser tab.

def df_window(df):
    with NamedTemporaryFile(delete=False, suffix='.html') as f:
        df.to_html(f)
    webbrowser.open(f.name)
    
def plot_snow(names, data):
    
    fig, axes = plt.subplots( nrows = 4, ncols = 4 )
    fig.suptitle('Fidelity Tests of Snow Patterns [m]', color = 'white')
    for p, dat, ax in zip( names, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = 0, vmax = 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p,fontsize = 8, color = 'white')
    
    # if # subplots is prime
    
    fig.delaxes(axes[-1,-1])
    fig.delaxes(axes[-1,-2])
    fig.delaxes(axes[-1,-3])

    
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [0,1,2] ) )
    cax.tick_params(labelsize = 8, colors = 'white')
    
    
def plot_tests(names, test_vals, test_name, data, rows, cols, cmin, cmax):
    
    fig, axes = plt.subplots( nrows = 4, ncols = 4 )
    fig.suptitle( test_name + 'Fidelity Test of Snow Patterns' )
    
    for p, v, dat, ax in zip( names, test_vals, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = cmin, vmax = cmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(p + " " + test_name + str(v), fontsize = 6, color = 'white' )
    
    # if # subplots is strange
    if len(names) != rows*cols:
        diff = -1*( rows*cols - len(names))
        i = -1
        while i >= diff:
            fig.delaxes(axes[-1,i])
            i = i-1
    
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [cmin, cmax] ) )
    cax.tick_params(labelsize = 6, colors = 'white')
    
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

#Snow

src1 = rasterio.open( '/home/cparr/Snow_Patterns/snow_data/happy_valley/raster/snow_on/hv_snow_watertrack_square2012.tif' )
snow_test = src1.read(1)
snow_test = snow_test.astype('float64')


# In[4]:

'''
Snow Data Test
'''

snow_data = []

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
mag_maps = []
freqs = []
dct_maps = []
dct_curves = []

# Create the test snows.
def warp_snow(snow):
    
    snow_data.append(snow)
    
    rows, cols = snow.shape
    mu = snow.mean()
    sigma = snow.std()

    # 90 degree rotation
    rotate90 = np.rot90(snow)
    snow_data.append(rotate90)
    
    #45 degree rotation
    oblique = rotate(snow, 45)
    b = oblique == 0
    oblique[b] = np.random.normal(mu, sigma, size=b.sum())
    snow_data.append(oblique)
    
    # morphological dilation and erosion

    selem = square(7)
    morph_dilation = dilation(snow, selem)
    morph_erosion = erosion(snow, selem)
    snow_data.append(morph_dilation)
    snow_data.append(morph_erosion)
    
    # flip up and down, basically a phase shift
    inverse = np.flipud(snow)
    snow_data.append(inverse)
    
    # a shift or translation
    shift_M = np.float32([[1,0,1],[0,1,0]])
    shifted = cv2.warpAffine(snow,shift_M,(cols,rows))
    snow_data.append(shifted)
    
    # randomly shuffle rows of array, create a random frequency
    permutation = np.random.permutation(snow)
    snow_data.append(permutation)
    
    # Random between bounds
    random_abs1 = np.random.uniform(snow.min(), snow.max(), [rows, cols])
    snow_data.append(random_abs1)
    
    # Gaussian noise
    mu = snow.mean()
    sigma = snow.std()
    gauss_abs1 = np.random.normal(mu, sigma, (rows, cols))
    snow_data.append(gauss_abs1)
    
    # Random Affine Transformation
    
    c = np.round(np.random.rand( 3,2 ), 2)
    m = np.append( c, ( 0,0,1 ) )
    m = m.reshape( 3,3 )
    aff_t = AffineTransform( matrix = m )
    random_aff_warp = warp( snow, aff_t )
    b = random_aff_warp == 0
    random_aff_warp[b] = np.random.normal(mu, sigma, size=b.sum())
    snow_data.append(random_aff_warp)
    
    # Additive Gaussian Noise
    noise = random_noise( snow, mode = 'gaussian' )
    snow_data.append( noise )
    
    # More Additive Gaussian Noise
    more_noise = random_noise(random_noise(random_noise(random_noise( noise, mode = 'gaussian' ))))
    snow_data.append( more_noise )
    
# Plot Titles and dictionary keys
snow_names = ['Reference', 'Rotate 90', 'Rotate 45', 'Dilation',
                 'Erosion', 'Y - Reflection', 'X Shift', 'Row Shuffle', 'Random', 'Gauss',
                 'Random Affine', 'Add Gaussian Noise','More Noise']

# Call It.
warp_snow( snow_test )

# Call Metrics on list of test snows

structural_sim( snow_data )
reg_mse( snow_data )
procrustes_analysis( snow_data )
make_quadrants( snow_data )

imse(snow_data)

cw_ssim_value(snow_data, 30)

transform( snow_data )
disccost( snow_data )

# Zip names, data, metrics, quadrants into a mega list!
# Generally this is indavisable because it relies on indexing...in the next cell we will make a dictionary.
snow_zip = zip(snow_names,snow_data, mse_vals, ssim_vals, disp_vals, top_lefts, top_rights, low_lefts, low_rights, 
               imse_vals, imse_maps, mse_maps, ssim_maps, cw_ssim_vals, cw_ssim_maps, mag_maps, freqs, dct_maps, dct_curves )


# In[5]:

snow_dict = defaultdict(dict)

'''
# Making a look up dictionary from all the patterns and their comparison scores.
# zipped list [i][0] is namne, 1 is full array, 2 is mse val, 3 is SSIM, 4 is PD,
# 5 through 8 are quadrants, 9 is IMSE, 10 is IMSE Map, 11 is MSE Map, 12 SSIM Map, 13 CWSSIM, 14 CWSSIM Map
'''

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
        data_dict[keys[i]]['Mag. Map'] = data_zip[i][15]
        data_dict[keys[i]]['Frequency Transects'] = data_zip[i][16]
        data_dict[keys[i]]['DCT Map'] = data_zip[i][17]
        data_dict[keys[i]]['DCT Curve'] = data_zip[i][18]

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


# In[6]:

to_dict_w_hists( snow_dict, snow_names, snow_zip )
snow_df = pd.DataFrame.from_dict(snow_dict)
snow_df = snow_df.transpose()


# In[7]:

# Histogram Scores

hist_scores = snow_df.loc[:,['name', 'Bhattacharyya UL','Bhattacharyya UR','Bhattacharyya LL',
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


# In[8]:

# Snow Scores and Ranks

snow_scores = snow_df.copy()
snow_scores['Pattern'] = snow_df['name']
snow_scores = snow_scores[['Pattern', 'MSE', 'SSIM', 'IMSE', 'Procrustres Disparity', 'CW SSIM']]
snow_scores = snow_scores.sort_values( 'CW SSIM', ascending = False )

ranks = snow_scores.copy()
ranks['Pattern'] = snow_df['name']
ranks['MSE Rank'] = np.round(snow_scores['MSE'].rank(ascending=True))
ranks['SSIM Rank'] = snow_scores['SSIM'].rank(ascending=False)
ranks['IMSE Rank'] = np.round(snow_scores['IMSE'].rank(ascending=True))
ranks['CW-SSIM Rank'] = snow_scores['CW SSIM'].rank(ascending=False)
ranks['Disparity Rank'] = snow_scores['Procrustres Disparity'].rank()
ranks['Bhattacharyya Rank'] = hist_scores['Mean Bhattacharyya'].rank(ascending=True)
ranks['Chi Square Rank'] = hist_scores['Mean Chi Square'].rank(ascending=True)
ranks['Correlation Rank'] = hist_scores['Mean Correlation'].rank(ascending=False)
del ranks['MSE']
del ranks['IMSE']
del ranks['SSIM']
del ranks['CW SSIM']
del ranks ['Procrustres Disparity']
ranks = ranks.sort_values('CW-SSIM Rank')

#df_window(snow_scores)


# In[24]:

render_mpl_table(ranks)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/snow_test_ranks.png', bbox_inches = 'tight', dpi = 300)
plt.close()

render_mpl_table(snow_scores)
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/snow_test_scores.png', bbox_inches = 'tight', dpi = 300)
plt.close()

plot_snow( snow_names, snow_data )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_snow_warps.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()
# names, test_vals, test_name, data, rows, cols, cmin, cmax

plot_tests( snow_names, imse_vals, " IMSE: ", imse_maps, 4, 4, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_imse_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, mse_vals, " MSE: ", mse_maps, 4, 4, 0, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_mse_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, ssim_vals, " SSIM: ", ssim_maps, 4, 4, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_ssim_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()

plot_tests( snow_names, cw_ssim_vals, " CW SSIM: ", cw_ssim_maps, 4, 4, -1, 1 )
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_cw_ssim_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
plt.close()


# In[25]:

from skimage import measure


fig, axes = plt.subplots( nrows = 4, ncols = 4 )
fig.suptitle('Fidelity Tests of Snow Depth Patterns [m]', color = 'white')
for p, dat, ax in zip( snow_names, snow_data, axes.flat ):
    
    contours = measure.find_contours(dat, 0.8)

    # The vmin and vmax arguments specify the color limits
    im = ax.imshow(dat, cmap = 'gray', interpolation = 'nearest', vmin = 0, vmax = 2)
    
    for n, contour in enumerate(contours):
        if contour.size >= 150:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5)

    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(p,fontsize = 8, color = 'white')
    ax.axis('image')


# if # subplots is prime

fig.delaxes(axes[-1,-1])
fig.delaxes(axes[-1,-2])
fig.delaxes(axes[-1,-3])


# Make an axis for the colorbar on the bottom

cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
fig.colorbar( im, cax=cax, ticks = ( [0,1,2] ) )
cax.tick_params(labelsize = 8, colors = 'white')

plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_contour_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')


# In[12]:

def plot_transforms(names, data, title, rows, cols, cmin, cmax):
    
    fig, axes = plt.subplots( nrows = rows, ncols = cols )
    fig.suptitle( title )
    
    for p, dat, ax in zip( names, data, axes.flat ):
        # The vmin and vmax arguments specify the color limits
        im = ax.imshow(dat, cmap = 'viridis', interpolation = 'nearest', vmin = cmin, vmax = cmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title( p, fontsize = 8 )
    
    # if # subplots is strange
    if len(names) != rows*cols:
        diff = -1*( rows*cols - len(names))
        i = -1
        while i >= diff:
            fig.delaxes(axes[-1,i])
            i = i-1
    
    # Make an axis for the colorbar on the bottom
    
    cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
    fig.colorbar( im, cax=cax, ticks = ( [cmin, cmax] ) )
    cax.tick_params(labelsize = 8)


# In[23]:

### transform plots

# plot_transforms( snow_names, mag_maps, "FFT Maps of Snow Patterns", 4, 4, 0, mag_maps[0].max() )
# plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_fft_map.png', bbox_inches = 'tight',
#             dpi = 300)
# plt.close()

# plot_transforms( snow_names, dct_maps, "DCT Maps of Snow Patterns", 4, 4, -5, 5)
# plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_dct_maps.png', bbox_inches = 'tight',
#             dpi = 300)
# plt.close()


fig, axes = plt.subplots( nrows = 4, ncols = 4 )

fig.suptitle( 'X Transect Mean of DCT of Snow Patterns' )

for p, dat, ax in zip( snow_names, dct_curves, axes.flat ):
    
    ymin = np.round(dat.min(),1)
    ymax = np.round(dat.max(),1)
    
    f = ax.plot( dat, lw = 1, color = '#236192' )
    ax.plot( dct_curves[0], lw = 1, ls = 'dashed', color = '#FFCD00', alpha = 0.67 )
    
    ax.set_yticks([ymin,ymax])
    ax.set_xticks([0,512])
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_yticklabels([ymin,ymax], size = 6)
    ax.set_xticklabels([0,512], size = 6)
    
    ax.set_xlim(0,512)
    ax.set_title( p, size = 7 )
    
# if # subplots is strange
if len(snow_names) != 16:
    diff = -1*( 16 - len(snow_names))
    i = -1
    while i >= diff:
        fig.delaxes(axes[-1,i])
        i = i-1
    
plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_dct_lines.png', bbox_inches = 'tight',
            dpi = 300)
plt.close()


# In[ ]:



