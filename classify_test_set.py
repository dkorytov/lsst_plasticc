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
    
class TimeSeriesReader:
    def __init__(self, location='data'):
        self.time_series_file = csv.reader(open(location+'/test_set.csv'), delimiter=',')
        self.header_line = next(self.time_series_file)
        self.header_dtypes = [np.int, np.float, np.int, np.float, np.float, np.bool]
        print(self.header_line)
        self.line_buffer = []

    def read_time_series_number(self, number, return_format="pandas", verbose=False):
        number_read = 0
        keep_reading = True
        read_line = True
        current_obj_id = 0
        while(keep_reading):
            if(read_line):
                line = next(self.time_series_file)
            else:
                read_line = True
            ts_obj_id = np.int64(line[0])
            if ts_obj_id != current_obj_id:
                if number_read >= number:
                    self.line_buffer.append(line)
                else:
                    number_read +=1
                    
        
    def read_time_series(self, df_metadata, return_format = "pandas", verbose=False):

        # We will stop reading the file once we read a line that has an 
        # object id larger than what's in reference df_metadata
        min_obj_id = np.min(df_metadata['object_id'])
        max_obj_id = np.max(df_metadata['object_id'])
        obj_id_list = np.array(df_metadata['object_id'])
        i = 0
        current_obj_id = obj_id_list[i]
        current_obj_id_seen = False
        print("id min: {} max: {}".format(min_obj_id, max_obj_id))

        keep_reading = True
        read_line = True
        while(keep_reading):
            if read_line:
                line = next(self.time_series_file)
                if verbose:
                    print("read line")
            else:
                read_line = True
                if verbose:
                    print("prev line (no read)")
            ts_obj_id = np.int64(line[0])
            if ts_obj_id == current_obj_id:
                self.line_buffer.append(line)
                current_obj_id_seen = True
                read_line = True
                if verbose:
                    print("reading current obj", ts_obj_id, max_obj_id)
            else:
                if i == (obj_id_list.size-1):
                    if verbose:
                        print("if i == (obj_id_list.size-1)")
                    keep_reading= False
                    next_read_line = line
                elif current_obj_id_seen:
                    if verbose:
                        print("if current_obj_id_seen")
                    i+=1
                    current_obj_id = obj_id_list[i]
                    read_line = False
                    current_obj_id_seen = False
                else:
                    if verbose:
                        print("else obj id", ts_obj_id)
                    pass
                
        # Depending on the return format requested
        if return_format == "pandas":
            csv_buffer = StringIO.StringIO()
            # Write the header
            [ csv_buffer.write(word+",") for word in self.header_line ]
            csv_buffer.write("\n")
            for line in self.line_buffer:
                [csv_buffer.write(word+",") for word in line]
                # Move the write cursor ontop of the last "," to overwrite it
                csv_buffer.seek(csv_buffer.tell()-1)
                # write a newline on top of ","
                csv_buffer.write("\n")
            # Reset the file cursor back to begining. As if we are reading
            # it for the first time.
            csv_buffer.seek(0)
            # Reset the line buffer for next time and add the line for the next obj
            # which is beyond the range we wanted for this read
            self.line_buffer = [next_read_line]
            return pd.read_csv(csv_buffer)
        elif return_format == "dict":
            result_dict = {}

            column_major = np.transpose(self.line_buffer)
            for i, (name, dtype) in enumerate(zip(self.header_line, self.header_dtypes)):
                result_dict[name] = np.array(column_major[i], dtype=dtype)
            self.line_buffer = [next_read_line]
            return result_dict
        elif return_format == "none":
            return None
        else:
            raise ValueError("return_format must either be \"pandas\" or \"dict\"")
            


def test():
    ts_reader = TimeSeriesReader()
    chunks = pd.read_csv("data/test_set_metadata.csv", chunksize = 2000)
    t1 = time.time()
    for i, df_metadata in enumerate(chunks):
        print("\n", i, df_metadata.shape)
        df_ts = ts_reader.read_time_series(df_metadata, return_format='none', verbose=False)
        t2 = time.time()
        print(i, t2-t1)
        t1 = t2

def test2():
    ts_reader = TimeSeriesReader()
    df_metadata = pd.read_csv("data/test_set_metadata.csv")
    

def test_ts_order():
    reader = csv.reader(open('data/test_set.csv'), delimiter=',')
    max_seen_obj_id = 0
    next(reader)
    num = 0
    for i, line in enumerate(reader):
        new_id = np.int64(line[0])

        if new_id > max_seen_obj_id:
            max_seen_obj_id = new_id
            print("new {} {} {:.2f}".format(i, max_seen_obj_id, np.log10(max_seen_obj_id)))
        elif new_id < max_seen_obj_id:
            print("{} < {} = {}".format(new_id,  max_seen_obj_id, (new_id > max_seen_obj_id)))
            print(new_id < max_seen_obj_id)
            print("i: ", i)
            print("max: ", max_seen_obj_id)
            print("new: ", new_id)
            max_seen_obj_id = new_id
            num += 1
            tmp = input()
        else:
            #print(i, new_id)
            pass
    print("bad values: ", num)

    
if __name__ == "__main__":
    test2()
#    tmp = input()
#    test_ts_order()
            



