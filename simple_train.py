#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sklearn as skl


def max_above_mean(data):
    return np.max(data)-np.average(data)

def min_below_mean(data):
    return np.min(data)-np.average(data)

def calc_mean(data):
    return data.mean()

def calc_prop(hmm):
    pass
    # print("\n")
    # print(type(hmm))
    # print(hmm.keys())
    # print(hmm)
    # result = {}
    # result['mean'] = np.mean(data['flux'])
    # result['max_above'] = np.max(data['flux'])-result['mean']
    # result['min_below'] = np.min(data['flux'])-result['mean']
    # result['variance']  = np.std(data['flux'])
    #df = pd.DataFrame(result)



    
if __name__ == "__main__":
    train_ts = pd.read_csv("data/training_set.csv")
    train_md = pd.read_csv("data/training_set_metadata.csv")
    types = np.sort(train_md['target'].unique())
    
    object_ids = train_md['object_id']
    
    print(train_ts.keys())
    print(train_md.keys())
    print('+++++++++++++++++++++++++')
    pass_bands = [0,1,2,3,4,5]

    cmap = plt.get_cmap('tab10')

    train_ts = train_ts.drop(['flux_err', 'detected', 'mjd'], axis = 1)
    train_grouped = train_ts.groupby(['object_id', 'passband'])
    print(train_ts.head(10))
    print(train_ts.groupby(['object_id', 'passband']).mean())
    exit()
    train_test2 = train_ts.groupby(['object_id', 'passband']).mean().reset_index()

    train_test = train_grouped.agg([np.mean, max_above_mean, min_below_mean])
    # print(train_test.apply('flux'.join).reset_index())
    # print("===")
    # print(train_test.head(10))
    # print("===")
    print(train_test2.head(10))
    print(train_test2.pivot(index='object_id', columns='passband', values='flux').head())
    
    # for obj_type in types:
    #     print("===object type: {}".format(obj_type))
    #     object_ids = train_md[train_md['target']==obj_type]['object_id'].sample(9)
    #     fig, axs = plt.subplots(3,3,figsize=(15,10))
    #     for i, obj_id in enumerate(object_ids):
    #         ax = axs[int(i/3)][i%3]
    #         obj_ts = train_ts[train_ts['object_id']==obj_id]
    #         for j, pass_band in enumerate(pass_bands):
    #             color = cmap(j)
    #             obj_band_ts = obj_ts[obj_ts['passband']==pass_band]
    #             slct_detected = obj_band_ts['detected']==1
    #             ax.scatter(obj_band_ts['mjd'][slct_detected], obj_band_ts['flux'][slct_detected],
    #                        label="{}".format(pass_band), edgecolor=color)
    #             ax.scatter(obj_band_ts['mjd'][~slct_detected], obj_band_ts['flux'][~slct_detected],
    #                        edgecolor=color, facecolor='none',label=None)
    #             #ax.fill_between(obj_band_ts['mjd'], obj_band_ts['flux']-obj_band_ts['flux_err'], obj_band_ts['flux']+obj_band_ts['flux_err'], alpha=0.3)
    #             ax.fill_between(obj_band_ts['mjd'], obj_band_ts['flux_err'], -obj_band_ts['flux_err'], alpha=0.3, color = color)
    #             ax.set_xlim([59500, 61000])
    #         plt.legend(loc='best')
    #     fig.suptitle("Class {}".format(obj_type))
    #     fig.tight_layout()
    # print("now showing")
    # plt.show()
