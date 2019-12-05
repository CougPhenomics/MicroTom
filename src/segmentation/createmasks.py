from plantcv import plantcv as pcv
import os
import numpy as np
import cv2 as cv2
from skimage import filters
from skimage import morphology



# %%


# def get_npq_frame(fn):
#     '''

#     '''
#     fns = os.listdir(indir)
#     import re
#     new_list = [x for x in fns if re.search(basefn, x)]
    
#     df3 = df2.query('jobdate == "2019-05-06"')
#     maxid = df3.imageid.max()
#     npqdf = df3.query('imageid == @maxid')
#     fmp,_,_ = pcv.readimage(npqdf.filename[0])
#     npq = np.divide(img,fmp) - 1
#     pcv.plot_image(npq)
#     np.max(npq[!np.where(np.isnan(npq))])
#     npq[!isnan(npq)].max()


# %%
def psIImask(img, mode='thresh'):
    # pcv.plot_image(img)
    if mode is 'thresh':

        mask = pcv.threshold.otsu(img, 255, 'light')
        # this entropy based technique seems to work well when algae is present
        # algaethresh = filters.threshold_yen(image=img)
        # threshy = pcv.threshold.binary(img, algaethresh, 255, 'light')
        # mask = pcv.dilate(threshy, 2, 1)
        mask = pcv.fill(mask, 100)
        # mask = pcv.erode(mask, 2, 2)
        final_mask = mask  # pcv.fill(mask, 270)

    elif isinstance(mode, pd.DataFrame):
        mode = curvedf
        rownum = mode.imageid.values.argmax()
        imgdf = mode.iloc[[1,rownum]]
        fm = cv2.imread(imgdf.filename[0])
        fmp = cv2.imread(imgdf.filename[1])        
        npq = np.float32(np.divide(fm,fmp, where = fmp != 0) - 1)
        npq = np.ma.array(fmp, mask = fmp < 200)
        plt.imshow(npq)
        # pcv.plot_image(npq)

        final_mask = np.zeroes(np.shape(img))

    else:
        pcv.fatal_error('mode must be "thresh" (default) or "npq")')

    return final_mask


# %%
