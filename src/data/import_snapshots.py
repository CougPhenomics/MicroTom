# -*- coding: utf-8 -*-
import os
import glob
import re as re
from datetime import datetime, timedelta
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

    fns = [fn for fn in glob.glob(pathname=os.path.join(snapshotdir,'*.tif'))]
    fns

    flist = list()
    for fn in fns:
        f=re.split('[-\\ ]', os.path.splitext(os.path.basename(fn))[0])
        f.append(fn)
        flist.append(f)

    fdf=pd.DataFrame(flist,columns=['sampleid','experiment','year','month','day','time','imageid','filename'])
    fdf[['year','month','day']] = fdf[['year','month','day']].astype('int16')

    # convert date and time columns to datetime format
    fdf['date'] = pd.to_datetime(fdf.loc[:,['year','month','day']])
    fdf['time'] = pd.to_timedelta(fdf.time.str.replace('_',':'))
    fdf['datetime'] = fdf.date + fdf.time
    fdf['jobdate'] = fdf['date']

    if camera.upper() == 'PSII':
        #create a jobdate to match dark and light measurements. dark experiments after 8PM correspond to the next day's light experiments
        fdf.loc[fdf.datetime.dt.hour >= 20,'jobdate'] = fdf.loc[fdf.datetime.dt.hour >= 20,'date'] + timedelta(days=1)

        # convert image id from string to integer that can be sorted numerically
        fdf['imageid'] = fdf.imageid.str.rsplit('_',-1).str[1].astype('uint8')
        fdf = fdf.sort_values(['sampleid','datetime','imageid'])


    fdf = fdf.set_index(['sampleid','experiment','datetime','jobdate']).drop(columns=['year','month','day','date','time'])

    # check for duplicate jobs of the same sample on the same day.  if jobs_removed.csv isnt blank then you shyould investigate!
    #dups = fdf.reset_index('datetime',drop=False).set_index(['imageid'],append=True).index.duplicated(keep='first')
    #dups_to_remove = fdf[dups].drop(columns=['imageid','filename']).reset_index().drop_duplicates()
    #dups_to_remove.to_csv('jobs_removed.csv',sep='\t')
    #

    return fdf
