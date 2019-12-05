# -*- coding: utf-8 -*-
import os
import glob
import re as re
from datetime import timedelta
import pandas as pd

def import_snapshots(snapshotdir, camera='vis'):
    '''
    Input:
    snapshotdir = directory of .tif files
    camera = the camera which captured the images. 'vis' or 'psii'

    Export .tif into snapshotdir from LemnaBase using format {0}-{3}-{1}-{6}
    '''

    # %% Get metadata from .tifs
    # snapshotdir = 'data/raw_snapshots/psII'

    fns = [fn for fn in glob.glob(pathname=os.path.join(snapshotdir,'*.png'))]
    fns

    flist = list()
    for fn in fns:
        f=re.split('[_-]', os.path.splitext(os.path.basename(fn))[0])
        f.append(fn)
        flist.append(f)

    fdf=pd.DataFrame(flist,columns=['sampleid','experiment','timestamp','cameralabel','imageid','filename'])

    # convert date and time columns to datetime format
    fdf['datetime'] = pd.to_datetime(fdf['timestamp'])
    fdf['jobdate'] = fdf.datetime.dt.floor('d')

    if camera.upper() == 'PSII':
        #create a jobdate to match dark and light measurements. dark experiments after 8PM correspond to the next day's light experiments
        fdf.loc[fdf.datetime.dt.hour >= 20,'jobdate'] = fdf.loc[fdf.datetime.dt.hour >= 20,'jobdate'] + timedelta(days=1)

        # convert image id from string to integer that can be sorted numerically
        fdf['imageid'] = fdf.imageid.astype('uint8')
        fdf = fdf.sort_values(['sampleid','datetime','imageid'])

    fdf = fdf.set_index(['sampleid','experiment','datetime','jobdate']).drop(columns = ['timestamp'])

    # check for duplicate jobs of the same sample on the same day.  if jobs_removed.csv isnt blank then you shyould investigate!
    #dups = fdf.reset_index('datetime',drop=False).set_index(['imageid'],append=True).index.duplicated(keep='first')
    #dups_to_remove = fdf[dups].drop(columns=['imageid','filename']).reset_index().drop_duplicates()
    #dups_to_remove.to_csv('jobs_removed.csv',sep='\t')
    #

    return fdf
