#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sklearn as skl



if __name__ == "__main__":
    train_ts = pd.read_csv("data/training_set.csv")
    train_md = pd.read_csv("data/training_set_metadata.csv")
    types = np.sort(train_md['target'].unique())
    
    object_ids = train_md['object_id']
    
    print(train_ts.sample(15))
    print(train_md.sample(15))

    pass_bands = [0,1,2,3,4,5]
    # for pass_band in pass_bands:
    #     plt.figure()
    #     slct = train_ts['passband']==pass_band
    #     h, xbins, ybins =np.histogram2d(train_ts['mjd'][slct], train_ts['flux_err'][slct], bins = (256, np.linspace(0,200,256)))
    #     plt.pcolor(xbins,ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    #     plt.title(str(pass_band))
    # plt.show()
    for obj_type in types:
        print("===object type: {}".format(obj_type))
        object_ids = train_md[train_md['target']==obj_type]['object_id'].sample(9)
        fig, axs = plt.subplots(3,3,figsize=(15,10))
        for i, obj_id in enumerate(object_ids):
            ax = axs[int(i/3)][i%3]
            obj_ts = train_ts[train_ts['object_id']==obj_id]
            for pass_band in pass_bands:
                obj_band_ts = obj_ts[obj_ts['passband']==pass_band]
                ax.plot(obj_band_ts['mjd'], obj_band_ts['flux'], 'o', label=pass_band)
                slct_detected = obj_band_ts['detected']==0
                #ax.plot(obj_band_ts['mjd']
                #ax.fill_between(obj_band_ts['mjd'], obj_band_ts['flux']-obj_band_ts['flux_err'], obj_band_ts['flux']+obj_band_ts['flux_err'], alpha=0.3)
                ax.fill_between(obj_band_ts['mjd'], obj_band_ts['flux_err'], -obj_band_ts['flux_err'], alpha=0.3)
                ax.set_xlim([59500, 61000])
            plt.legend(loc='best')
        fig.suptitle("Class {}".format(obj_type))
        fig.tight_layout()
    print("now showing")
    plt.show()
