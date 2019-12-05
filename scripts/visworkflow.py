#!python

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import re
import numpy as np
import argparse
import cv2
from plantcv import plantcv as pcv
import os
from matplotlib import pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(
        description="Imaging processing with opencv")
    parser.add_argument("-i",
                        "--image",
                        help="Input image file.",
                        required=True)
    parser.add_argument("-o",
                        "--outdir",
                        help="Output directory for image files.",
                        required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w",
                        "--writeimg",
                        help="write out images.",
                        default=False,
                        action="store_true")
    parser.add_argument(
        "-D",
        "--debug",
        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
        default=None)
    parser.add_argument(
        "--debugdir",
        help="Directory for debuging images",
        required=False)
    parser.add_argument(
        "--regex",
        help="Format to parse filename into metadata",
        required=False
    )
    args = parser.parse_args()
    return args


def vismask(img):

    a_img = pcv.rgb2gray_lab(img, channel='a')
    # thresh_a = pcv.threshold.binary(a_img, 124, 255, 'dark')
    thresh_o = pcv.threshold.otsu(a_img, 255, 'dark')
    # b_img = pcv.rgb2gray_lab(img, channel='b')
    # thresh_b = pcv.threshold.binary(b_img, 127, 255, 'light')

    # mask = pcv.logical_and(thresh_a, thresh_b)
    mask = pcv.fill(thresh_o, 200)
    final_mask = pcv.dilate(mask, 2, 1)

    return final_mask


# Main workflow
def main():
    # Get options
    args = options()

    if args.debug:
        pcv.params.debug = args.debug  # set debug mode
        if args.debugdir:
            pcv.params.debug_outdir = args.debugdir  # set debug directory
            os.makedirs(args.debugdir, exist_ok=True)

    # pixel_resolution
    # mm
    # see pixel_resolution.xlsx for calibration curve for pixel to mm translation
    pixelresolution = 0.055
    # plt.rcParams["font.family"] = "Arial"  # All text is Arial

    # The result file should exist if plantcv-workflow.py was run
    if os.path.exists(args.result):
        # Open the result file
        results = open(args.result, "r")
        # The result file would have image metadata in it from plantcv-workflow.py, read it into memory
        metadata = json.load(results)
        # Close the file
        results.close()
        # Delete the file, we will create new ones
        os.remove(args.result)
        plantbarcode = metadata['metadata']['plantbarcode']['value']
        print(plantbarcode,
              metadata['metadata']['timestamp']['value'], sep=' - ')

    else:
        # If the file did not exist (for testing), initialize metadata as an empty string
        metadata = "{}"
        regpat = re.compile(args.regex)
        plantbarcode = re.search(regpat, args.image).groups()[0]

    # read images and create mask
    img, _, fn = pcv.readimage(args.image)
    imagename = os.path.splitext(fn)[0]
    mask = vismask(img)
    final_mask = np.zeros_like(mask)

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

    # greenness index
    out_flt = np.zeros_like(denom, dtype='float32')
    # divide green by sum of channels to compute greenness index with values 0-1
    gi = np.divide(g, denom, out=out_flt,
                   where=np.logical_and(denom != 0, mask > 0))

    # find objects
    c, h = pcv.find_objects(img, mask)
    roi_c, roi_h = pcv.roi.multi(img,
                                 coord=(1250, 1800),
                                 radius=400,
                                 spacing=(0, 0),
                                 ncols=1,
                                 nrows=1)

    # Turn off debug temporarily, otherwise there will be a lot of plots
    pcv.params.debug = None
    # Loop over each region of interest
    i=0
    rc_i = roi_c[i]
    for i, rc_i in enumerate(roi_c):
        rh_i = roi_h[i]

        # Add ROI number to output. Before roi_objects so result has NA if no object.
        pcv.outputs.add_observation(
            variable='roi',
            trait='roi',
            method='roi',
            scale='int',
            datatype=int,
            value=i,
            label='#')
        
        roi_obj, hierarchy_obj, submask, obj_area = pcv.roi_objects(
            img, roi_contour=rc_i, roi_hierarchy=rh_i, object_contour=c, obj_hierarchy=h, roi_type='partial')
        
        if obj_area == 0:
            
            print('\t!!! No object found in ROI', str(i))
            pcv.outputs.add_observation(
                variable='plantarea',
                trait='plant area in sq mm',
                method='observations.area*pixelresolution^2',
                scale=pixelresolution,
                datatype="<class 'float'>",
                value=0,
                label='sq mm')

        else:
 
            # Combine multiple objects
            # ple plant objects within an roi together
            plant_object, plant_mask = pcv.object_composition(
                img=img, contours=roi_obj, hierarchy=hierarchy_obj)

            final_mask = pcv.image_add(final_mask, plant_mask)

            # Save greenness for individual ROI
            grnindex = np.mean(gi[np.where(plant_mask > 0)])
            pcv.outputs.add_observation(
                variable='greenness_index',
                trait='mean normalized greenness index',
                method='g/sum(b+g+r)',
                scale='[0,1]',
                datatype="<class 'float'>",
                value=float(grnindex),
                label='/1')
        
            # Analyze all colors
            hist = pcv.analyze_color(img, plant_mask, 'all')
            
            # Analyze the shape of the current plant
            shape_img = pcv.analyze_object(img, plant_object, plant_mask)
            plant_area = pcv.outputs.observations['area']['value'] * pixelresolution**2
            pcv.outputs.add_observation(
                variable='plantarea',
                trait='plant area in sq mm',
                method='observations.area*pixelresolution^2',
                scale=pixelresolution,
                datatype="<class 'float'>",
                value=plant_area,
                label='sq mm')
            
        # end if-else

        # At this point we have observations for one plant
        # We can write these out to a unique results file
        # Here I will name the results file with the ROI ID combined with the original result filename
        basename, ext = os.path.splitext(args.result)
        filename = basename + "_" + str(i) + ext
        # Save the existing metadata to the new file
        with open(filename, "w") as r:
            json.dump(metadata, r)
        pcv.print_results(filename=filename)
        # The results are saved, now clear out the observations so the next loop adds new ones for the next plant
        pcv.outputs.clear()

        if args.writeimg and obj_area != 0:
            imgdir = os.path.join(args.outdir, 'shape_images', plantbarcode)
            os.makedirs(imgdir, exist_ok=True)
            pcv.print_image(shape_img, os.path.join(imgdir, imagename + '_' + str(i) + '_shape.png'))

            imgdir = os.path.join(args.outdir, 'colorhist_images', plantbarcode)
            os.makedirs(imgdir, exist_ok=True)
            pcv.print_image(hist, os.path.join(imgdir, imagename + '_' + str(i) + '_colorhist.png'))

    # end roi loop

    if args.writeimg:
        # save grnness image of entire tray
        imgdir = os.path.join(args.outdir, 'pseudocolor_images', plantbarcode)
        os.makedirs(imgdir, exist_ok=True)
        gi_img = pcv.visualize.pseudocolor(
            gi, obj=None, mask=final_mask, cmap='viridis', axes=False, min_value=0.3, max_value=0.5, background='black', obj_padding=0)
        gi_img = add_scalebar(
            gi_img, pixelresolution=pixelresolution, barwidth=20, barlocation='lower left')
        gi_img.set_size_inches(6, 6, forward=False)
        gi_img.savefig(os.path.join(imgdir, imagename + '_greenness.png'), bbox_inches='tight')
        gi_img.clf()

# end of function!


def add_scalebar(pseudoimg, pixelresolution, barwidth, barlocation='lower center', fontprops=None, scalebar=None):
    if fontprops is None:
        fontprops = fm.FontProperties(size=16, weight='bold')

    ax = pseudoimg.gca()

    if scalebar is None:
        scalebar = AnchoredSizeBar(ax.transData,
                                   barwidth/pixelresolution,  '2 cm', barlocation,
                                   pad=0.5,
                                   sep=5,
                                   color='white',
                                   frameon=False,
                                   size_vertical=barwidth/pixelresolution/30,
                                   fontproperties=fontprops)

    ax.add_artist(scalebar)

    return ax.get_figure()


if __name__ == "__main__":
    main()
