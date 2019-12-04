# %% Setup
# Export .tif into outdir from LemnaBase using format {0}-{3}-{1}-{6}
from plantcv import plantcv as pcv
import importlib
import os
from datetime import datetime, timedelta
import cv2 as cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

# %% Import functions from src/ directory to get snaphots, create masks, and setup image classification
from src.data import import_snapshots
from src.segmentation import createmasks
from src.util import masked_stats
from src.viz import add_scalebar, custom_colormaps

# %% io directories
indir = os.path.join('data', 'raw_snapshots', 'vis')

outdir = os.path.join('output', 'vis')
debugdir = os.path.join('debug', 'vis')
maskdir = os.path.join(outdir, 'masks')
os.makedirs(outdir, exist_ok=True)
os.makedirs(debugdir, exist_ok=True)
os.makedirs(maskdir, exist_ok=True)

# %% pixel pixel_resolution
# mm
# see pixel_resolution.xlsx for calibration curve for pixel to mm translation
pixelresolution = 0.052

# %% Import tif file information based on the filenames. If extract_frames=True it will save each frame form the multiframe TIF to a separate file in data/pimframes/ with a numeric suffix
fdf = import_snapshots.import_snapshots(indir, camera='vis')

# %%  Keep just the VIS images
df = fdf.query('imageid == "VIS0"').copy()

# %% Setup Debug parmaeters
# pcv.params.debug can be 'plot', 'print', or 'None'. 'plot' is useful if you are testing your pipeline over a few samples so you can see each step.
pcv.params.debug = 'plot'  # 'print' #'plot', 'None'
# Figures will show 9x9inches which fits my monitor well.
plt.rcParams["figure.figsize"] = (9, 9)
plt.rcParams["font.family"] = "Arial"  # All text is Arial

# %% The main analysis function
# I like to reload my mask function to make sure it's the latest if I've been optimizing it
# importlib.reload(createmasks)

# This function takes a dataframe of metadata that was created above. We loop through each pair of images to compute photosynthetic parameters
def image_avg(fundf):

    fn = fundf.filename[0]
    outfn = os.path.splitext(os.path.basename(fn))[0]
    outfn_split = outfn.split('-')
    outfn = "-".join(outfn_split)
    basefn = "-".join(outfn_split[0:-1])
    sampleid = outfn_split[0]
    print(outfn)

    if pcv.params.debug == 'print':
        debug_outdir = os.path.join(debugdir, outfn)
        os.makedirs(debug_outdir, exist_ok=True)
        pcv.params.debug_outdir = debug_outdir

    # read images and create mask
    img, _, _ = pcv.readbayer(fn)
    mask = createmasks.vismask(img)

    # find objects
    c, h = pcv.find_objects(img, mask)
    roi_c, roi_h = pcv.roi.multi(img, 
                                coord=(900, 1200), 
                                radius=500, 
                                 spacing=(0, 1475),
                                ncols=2, 
                                nrows=1)

    # setup individual roi plant masks
    newmask = np.zeros_like(mask)

    # Make as many copies of incoming dataframe as there are ROIs
    outdf = fundf.copy()
    for i in range(0, len(roi_c)-1):
        # print(i)
        outdf = outdf.append(fundf)

    # Compute greenness
    # split color channels
    b, g, r = cv2.split(img)
    # print green intensity
    # g_img = pcv.visualize.pseudocolor(g, cmap='Greens', background='white', min_value=0, max_value=255, mask=mask, axes=False)

    # convert color channels to int16 so we can add them (values will be greater than 255 which is max of current uint8 format)
    g = g.astype('uint16')
    r = r.astype('uint16')
    b = b.astype('uint16')
    denom = g + r + b
    
    out_flt = np.zeros_like(denom, dtype='float32')
    # divide green by sum of channels to compute greenness index with values 0-1
    gi = np.divide(g, denom, out = out_flt, where=np.logical_and(denom != 0, mask > 0))

    # Initialize lists to store variables for each ROI and iterate
    gi_avg = []
    gi_std = []
    ithroi = []
    plantarea = []
    inbounds = []
    i = 1
    rc = roi_c[i]
    for i, rc in enumerate(roi_c):
        # Store iteration Number
        ithroi.append(int(i))
        # extract ith hierarchy
        rh = roi_h[i]

        # Filter objects based on being in the ROI
        try:
            roi_obj, hierarchy_obj, submask, obj_area = pcv.roi_objects(img, roi_contour=rc, roi_hierarchy=rh, object_contour=c, obj_hierarchy=h, roi_type='partial')
        except RuntimeError as err:
            print('!!!', err, str(i))
            gi_avg.append(np.nan)
            gi_std.append(np.nan)
            inbounds.append(np.nan)
            plantarea.append(0)

        else:

            # Combine multiple plant objects within an roi together
            plant_contour, plant_mask = pcv.object_composition(
                img=img, contours=roi_obj, hierarchy=hierarchy_obj)

            # Save object area for each ROI to list to output later
            plantarea.append(obj_area * pixelresolution**2)

            # combine plant masks after roi filter
            newmask = pcv.image_add(newmask, plant_mask)

            # Mask Greenness array with submask from within ROI
            gi_masked = np.ma.array(gi, mask=~plant_mask.astype('bool'))

            gi_avg.append(gi_masked.mean())
            gi_std.append(gi_masked.std())

            inbounds.append(pcv.within_frame(plant_mask))

        #end try-except-else
    # end roi loop

    # save mask of all plants to file after roi filter
    pcv.print_image(newmask, os.path.join(maskdir, outfn + '_mask.png'))

    # save grnness image of entire tray    
    imgdir = os.path.join(outdir, 'pseudocolor_images', sampleid)
    os.makedirs(imgdir, exist_ok=True)
    gi_img = pcv.visualize.pseudocolor(
        gi, obj=None, mask=newmask, cmap='viridis', axes=False, min_value=0.3, max_value=0.6, background='black', obj_padding=0)
    gi_img = add_scalebar.add_scalebar(
        gi_img, pixelresolution=pixelresolution, barwidth=20, barlocation='lower left')
    gi_img.set_size_inches(6, 6, forward=False)
    gi_img.savefig(os.path.join(imgdir, outfn + '_greenness.png'), bbox_inches='tight')
    gi_img.clf()
    
    # it seems plant area and gi_avg alone don't always give the right answer and especially with many decimals, presumeably there ae small independent objects that fall in one roi but not the other that change the object slightly. 
    rounded_avg = [round(n,3) for n in gi_avg]
    rounded_std = [round(n,3) for n in gi_std]
    isunique = not (rounded_avg.count(rounded_avg[0]) == len(gi_avg) and 
        rounded_std.count(rounded_std[0]) == len(gi_std))

    outdf['roi'] = ithroi
    outdf['grnindex_avg'] = gi_avg
    outdf['grnindex_std'] = gi_std
    outdf['plantarea'] = plantarea
    outdf['obj_in_frame'] = inbounds
    outdf['unique_roi'] = isunique

    return(outdf)
# end of function!


# %% Setup Debug parameters
#by default params.debug should be 'None' when you are ready to process all your images
pcv.params.debug = 'None'
# if you choose to print debug files to disk then remove the old ones first (if they exist)
if pcv.params.debug == 'print':
    import shutil
    shutil.rmtree(os.path.join(debugdir), ignore_errors=True)

# %% Testing dataframe
# If you need to test new function or threshold values you can subset your dataframe to analyze some images
# df2 = df.query('(sampleid == "A4" & jobdate == "2019-05-14") | (sampleid == "B7" & jobdate == "2019-05-08")')
# del df2
# fundf = df2
# del fundf
# # # fundf
# # end testing

# %% Process the files
# check for subsetted dataframe
if 'df2' not in globals():
    df2 = df
else:
    print('df2 already exists!')

# Each unique combination of treatment, sampleid, jobdate, parameter should result in exactly 2 rows in the dataframe that correspond to Fo/Fm or F'/Fm'
dfgrps = df2.groupby(['sampleid', 'experiment', 'jobdate', 'datetime'])

grplist = []
for grp, grpdf in dfgrps:
    # print(grp)#'%s ---' % (grp))
    grplist.append(image_avg(grpdf))
df_avg = pd.concat(grplist)

# %% Add genotype information
gtypeinfo = pd.read_csv(
    os.path.join('data', 'genotype_map.csv'), 
    skipinitialspace=True)
df_avg2 = (pd.merge(df_avg.reset_index(),
                    gtypeinfo,
                    on=['sampleid', 'roi'],
                    how='inner')
           )

# %% Write the tabular results to file!
(df_avg2.drop(['imageid', 'filename'], axis=1)
        .sort_values(['jobdate', 'sampleid'])
        .to_csv(os.path.join(outdir, 'output_vis_level0.csv'), na_rep='nan', float_format='%.4f', index=False)
 )
