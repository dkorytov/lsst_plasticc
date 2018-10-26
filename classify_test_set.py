#!/usr/bin/env python2.7

from __future__ import print_function, division


import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import sklearn as skl
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import sklearn.ensemble
import sklearn.feature_selection

import StringIO
import csv
import time
import gc

from summary_statistics import get_summary_statistic_function_dict, create_summary_statistics
from util import merge_time_series_summary_and_metadata, TimeSeriesReader, concatenate_dictionaries

total_objects = 3492891


def test2(model_fname="models/decision_tree.pckl", output_fname = "prediction/prediction.csv", read_number = 50, batch_merge = 1000):
    ts_reader = TimeSeriesReader(fname = "data/test_set.csv")
    metadata_df = pd.read_csv("data/test_set_metadata.csv")
    summary_functions = get_summary_statistic_function_dict()
    first_write = True
    model = pickle.load(open(model_fname))
    total_read = 0
    t0 = time.time()
    while(ts_reader.has_lines_to_read()):
        t1 = time.time()
        print("reading...")
        summary_dicts = []
        t_read = 0
        t_sum = 0
        for i in range(0, batch_merge):
            t_a = time.time()
            ts_dict = ts_reader.read_time_series_number(read_number,
                                                        return_format='dict',
                                                        verbose=False)
            t_b = time.time()
            summary_dicts.append(create_summary_statistics(ts_dict, summary_functions))
            t_c = time.time()
            t_read += t_b-t_a
            t_sum +=t_c-t_b
            total_read += read_number
        print('\t\t raw read: {:.4f}'.format(t_read))
        print('\t\t summary:  {:.4f}'.format(t_sum))
        t2a = time.time()
        print("\ttotal", t2a-t1)


        summary_dict = concatenate_dictionaries(summary_dicts)
        t2 = time.time()
        print("\t ", t2-t2a)
        t3 = time.time()

        summary_df = pd.DataFrame.from_dict(summary_dict)
        print(summary_df.head())
        print("merging...")
        x_df = merge_time_series_summary_and_metadata(summary_df, metadata_df)
        obj_id_ds = x_df[['object_id']]
        x_df.drop(columns=['object_id'], inplace=True)
        # print(x_df.keys())
        # print(x_df)
        t4 = time.time()
        print("\t ", t4-t3)
        predict_df = pd.DataFrame.from_dict({"class": model.predict(x_df)})
        result_df = pd.concat((obj_id_ds,predict_df),
                              axis='columns',
                              ignore_index = True)
        result_df.columns = ['object_id','class_pred']
        result_df.class_pred = result_df.class_pred.astype('category',
                                                           categories=[6, 15, 16, 42,
                                                                       52, 53, 62, 64,
                                                                       65, 67, 88, 90,
                                                                       92, 95, 99])
        # print(result_df['class_pred'])
        onehot = pd.get_dummies(result_df["class_pred"], prefix="class")
        # print(result_df.sample(10))
        final_res = pd.concat([obj_id_ds, onehot], axis=1)
        final_res.object_id = final_res.object_id.astype(np.int32)
        # print(final_res.sample(10))
        if first_write:
            final_res.to_csv(output_fname, mode='w', header=True, index = False)
            first_write = False
        else:
            final_res.to_csv(output_fname, mode='a', header=False, index = False)
        t6 = time.time()
        print("step time: {:.2f}".format(t6-t1))
        print("total time: {:.2f}".format(t6-t0))
        print("time per obj: {:.4f}".format((t6-t1)/(read_number * batch_merge)))
        print("progess: {:.3f}".format(total_read/total_objects))
        print("\n\n")
        #exit()
    
if __name__ == "__main__":
    test2()
            



