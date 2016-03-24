'''
This script reads snow depth geotiffs and preprocesses them
and stores results to a dict.
Dict structure is src_name (str)
                      snow with NaN mask (numpy array)
                      rasterio metadata
                      Gaussian blurred snow (numpy array)
                      Normalized (0 to 1) snow (numpy array)
'''
# Initialize empty dictionary

src_dict = {}

# Can use glob to pull all snow rasters in a certain folder
test = ['/home/cparr/water_tracks/sub_wt_2012.tif', '/home/cparr/water_tracks/sub_wt_2013.tif']

def read_gtiffs(tif_list):
    
    for f in tif_list:
        
        src_name = os.path.basename( f ).replace( '.tif', '_src' )
        src = rasterio.open(f)
        meta = src.meta
        
        arr_tif = np.ma.masked_values( src.read(1), src.nodata )
        blur_snow = cv2.GaussianBlur(arr_tif, (3,3), 0)
        norm_snow = maxabs_scale(blur_snow, axis=1, copy=True)

        
        src_dict[src_name] = arr_tif, meta, blur_snow, norm_snow
        #return src_dict
        
    
read_gtiffs(test)

i = 1
for k,v in src_dict.iteritems():
    plt.subplot(1,2,i)
    plt.imshow(v[3])
    i += 1
