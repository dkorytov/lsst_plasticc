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




class TimeSeriesReader:
    def __init__(self, fname='data/test_set.csv'):
        self.time_series_file = csv.reader(open(fname), delimiter=',')
        self.header_line = next(self.time_series_file)
        self.header_dtypes = [np.int, np.float, np.int, np.float, np.float, np.bool]
        print(self.header_line)
        self.line_buffer = []
        self.file_good = True


    def more_lines_to_read(self):
        return self.file_good
    

    def read_time_series_number(self, number, return_format="pandas", verbose=False):
        number_read = 0
        keep_reading = True
        read_line = True
        current_obj_id = 0
        while(keep_reading):
            try:
                line = next(self.time_series_file)
            except StopIteration as e:
                next_read_line = ""
                self.file_good = False
                break
            ts_obj_id = np.int64(line[0])
            # print("\treading line")
            # print("\tobj id: ts_obj_id", ts_obj_id)
            if ts_obj_id != current_obj_id:
                if number_read < number:
                    current_obj_id = ts_obj_id
                    self.line_buffer.append(line)
                    number_read +=1
                    # print("\t new obj!")
                else:
                    keep_reading = False
                    next_read_line = line
                    # print("\t exiting")
            else:
                self.line_buffer.append(line)
                # print("\t same obj")
        print("Number found: ", number_read)
        # Depending on the return format requested
        if return_format == "dict" or return_format == "pandas":
            result_dict = {}
            column_major = np.transpose(self.line_buffer)
            for i, (name, dtype) in enumerate(zip(self.header_line, self.header_dtypes)):
                result_dict[name] = np.array(column_major[i], dtype=dtype)
            self.line_buffer = [next_read_line]
            if return_format =="dict":
                return result_dict
            else:
                return pd.DataFrame.from_dict(result_dict)
        elif return_format == "none":
            self.line_buffer = [next_read_line]
            return None
        else:
            raise ValueError("return_format must either be \"pandas\", \"dict\" or \"none\"")
            



def test2():
    gc.enable()
    ts_reader = TimeSeriesReader(fname = "data/training_set.csv")
    df_metadata = pd.read_csav("data/test_set_metadata.csv")
    while(ts_reader.more_lines_to_read()):
        print("reading...")
        ts_dict = ts_reader.read_time_series_number(2000, return_format='dict', verbose=False)
        print("unqiue size: ", np.unique(ts_dict['object_id']).size)
        ts_df = pd.DataFrame.from_dict(ts_dict)
        ts_df_gb = ts_df.groupby(['object_id', 'passband'])
        ts_df_sum = ts_df_gb.
        gc.collect()

    
if __name__ == "__main__":
    test2()
#    tmp = input()
#    test_ts_order()
            



