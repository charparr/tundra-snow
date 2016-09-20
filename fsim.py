
from __future__ import division
import os
from skimage import io
from skimage.util import random_noise
from skimage.filters import scharr
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import phasepack

def input_data(path, filename):
    
    img_path = os.path.join(path, filename)
    img = io.imread(img_path)
    img = img[85:341,90:346]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, gray

def _preprocess( reference_image, blur_amount ):
    
   blur = cv2.GaussianBlur( reference_image,( blur_amount, blur_amount ), 0 )
   # can also downsample and average filter
#   noise = random_noise( random_noise( random_noise(reference_image,
#                                                    mode = "gaussian") ))
   return blur

inputs = input_data( '/home/cparr/Downloads/jpeg2000_db/db/', 'rapids.bmp' )
img = inputs[0]
dst = _preprocess( img, 25 )

r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
imgY = 0.299 * r + 0.587 * g + 0.114 * b
imgI = 0.596 * r - 0.275 * g - 0.321 * b
imgQ = 0.212 * r - 0.523 * g + 0.311 * b

r_d, g_d, b_d = dst[:,:,0], dst[:,:,1], dst[:,:,2]
dstY = 0.299 * r_d + 0.587 * g_d + 0.114 * b_d
dstI = 0.596 * r_d - 0.275 * g_d - 0.321 * b_d
dstQ = 0.212 * r_d - 0.523 * g_d + 0.311 * b_d

t1 = 0.85
t2 = 160
t3 = 200
t4 = 200

s_Q = ( 2*imgQ + dstQ + t4 )  / ( imgQ**2 + dstQ**2 + t4 )

s_I = ( 2*imgI + dstI + t3 )  / ( imgI**2 + dstI**2 + t3 )

pc1 = phasepack.phasecong(imgY, nscale = 4, norient = 4, minWaveLength = 6, mult = 2, sigmaOnf=0.55)
pc2 = phasepack.phasecong(dstY, nscale = 4, norient = 4, minWaveLength = 6, mult = 2, sigmaOnf=0.55)
pc1 = pc1[0]
pc2 = pc2[0]

s_PC = ( 2*pc1 + pc2 + t1 )  / ( pc1**2 + pc2**2 + t1 )

g1 = scharr( imgY )
g2 = scharr( dstY )
s_G = ( 2*g1 + g2 + t2 )  / ( g1**2 + g2**2 + t2 )

s_L = s_PC * s_G
s_C = s_I * s_Q

pcM = np.maximum(pc1,pc2)


fsim = round( np.nansum( s_L * pcM) / np.nansum(pcM), 3)

fsimc = round( np.nansum( s_L * s_C**0.3 * pcM) / np.nansum(pcM), 3)

print 'FSIM: ' + str(fsim)
print 'FSIMC: ' + str(fsimc)


fig, axes = plt.subplots( nrows = 2, ncols = 3 )
    
plt.subplot(231)
plt.imshow(img)
plt.title('Reference')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(dst, cmap = 'gray')
plt.title('Distorted')
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(pc1, cmap = 'gray')
plt.title('Ref. PC', size = 8)
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.imshow(pc2, cmap = 'gray')
plt.title('Dist. PC', size = 8)
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(s_L, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('FSIM: '+ str(fsim))

fig.delaxes(axes[-1,-1]) 

plt.savefig('/home/cparr/Snow_Patterns/figures/gsmd/fsim_rapids.png',
            bbox_inches = 'tight', dpi = 300, facecolor = 'skyblue')

