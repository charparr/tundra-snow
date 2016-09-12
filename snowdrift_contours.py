import matplotlib.pyplot as plt
from skimage import measure

fig, axes = plt.subplots( nrows = 4, ncols = 4 )
fig.suptitle('Fidelity Tests of Snow Depth Patterns [m]', color = 'white')
for p, dat, ax in zip( snow_names, snow_data, axes.flat ):
    
    contours = measure.find_contours(dat, 0.8)

    # The vmin and vmax arguments specify the color limits
    im = ax.imshow(dat, cmap = 'gray', interpolation = 'nearest', vmin = 0, vmax = 2)
    
    for n, contour in enumerate(contours):
        # 150 for contour size is arbitrary
        if contour.size >= 150:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5)

    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(p,fontsize = 8, color = 'white')
    ax.axis('image')


# if num subplots is prime

fig.delaxes(axes[-1,-1])
fig.delaxes(axes[-1,-2])
fig.delaxes(axes[-1,-3])


# Make an axis for the colorbar on the bottom

cax = fig.add_axes( [0.05, 0.2, 0.04, 0.6] )
fig.colorbar( im, cax=cax, ticks = ( [0,1,2] ) )
cax.tick_params(labelsize = 8, colors = 'white')

plt.savefig('/home/cparr/Snow_Patterns/figures/hv_snow_test/hv_contour_map.png', bbox_inches = 'tight', dpi = 300, facecolor = 'black')
