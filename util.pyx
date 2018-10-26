
from __future__ import print_function, division


import itertools
import pickle
import numpy as np
import pandas as pd
import csv

import sklearn as skl
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import sklearn.ensemble
import sklearn.feature_selection

    
def concatenate_dictionaries(dicts):
    result = {}
    for key in dicts[0].keys():
        column_list = []
        for dict_elem in dicts:
            column_list.append(dict_elem[key])
        result[key] = np.concatenate(column_list)
    return result

def max_above_mean(data):
    return np.max(data)-np.average(data)

def min_below_mean(data):
    return np.min(data)-np.average(data)

def frac_max_flux(data):
    return np.max(data)/np.sum(data)

def frac_min_flux(data):
    return np.min(data)/np.sum(data)

def time_seen(df, indx):
    #This is very slow...
    df2 = df.loc[indx]
    slct = df2['detected']==1
    if(np.sum(slct) != 0):
        df2 = df2[slct]
        result = np.max(df2['mjd'])-np.min(df2['mjd'])
    else:
        result =  np.max(df2['mjd'])-np.min(df2['mjd'])
    return result

def df_dt_max(g, df, indx):
    df2 = df.loc[indx]
    return np.max(np.gradient(df2['mjd'], g))

def df_dt_min(g, df, indx):
    df2 = df.loc[indx]
    return np.min(np.gradient(df2['mjd'], g))

def df_dt_mean(g, df, indx):
    df2 = df.loc[indx]
    return np.mean(np.gradient(df2['mjd'], g))


def get_processed_data(df_ts):
    """takes in the raw time series data frame and returns a dataframe with 
    summary statistics for each object

    :param train_t:s  bleh
    :return : pandas data frame

    """


    print("\n\nDrop time series data we aren't using yet\n")    
    #df_ts = df_ts.drop(['flux_err', 'detected', 'mjd'], axis = 1)
    print(df_ts.head(10))
    
    print("\n\nGroup the time series by object and passband.\nWe will use a single value per object and passband combination group\n")
    train_grouped = df_ts.groupby(['object_id', 'passband'])

    ltime_seen = lambda g: time_seen(df_ts, g.index)
    ldf_dt_min = lambda g: df_dt_min(g, df_ts, g.index)
    ldf_dt_max = lambda g: df_dt_max(g, df_ts, g.index)
    ldf_dt_mean= lambda g: df_dt_mean(g, df_ts, g.index)
        
    print("\n\nFor each group (id + passband), we will calculate the mean, the maximum/minimum flux above the mean, and standard deviation of the time series\n")
    df_ts_agg = train_grouped.agg({'flux':{'time_seen': ltime_seen, 'df_dt_min': ldf_dt_min, 'df_dt_max': ldf_dt_max, 'df_dt_mean': ldf_dt_mean, 'mean': np.mean, 'max_above': max_above_mean, 'min_below': min_below_mean, 'frac_max': frac_max_flux, 'frac_min': frac_min_flux }, 'detected':{'detected_frac': np.average}}).reset_index()

    print("column names:", df_ts_agg.keys())

    print("\n\nThe column names now have a hierarchical structure with two levels. \nLets get rid of it\n") 
    df_ts_agg.columns = ([col1 if col1 != "flux" else col2 for col1, col2 in df_ts_agg.columns])
    print("column names: ", df_ts_agg.keys())

    print("\n\nLets use pivot to create a single row per object, with multiple entries per band and time serties property\n")
    df_ts_agg = df_ts_agg.pivot(index='object_id', columns='passband')#, values=['mean', 'max_above_mean', 'min_below_mean', 'frac_max_flux', 'frac_min_flux', 'ltime_seen'])
    print("Column names: ", df_ts_agg.keys())
    
    print("\n\nAgain, lets get rid of the hierarchical columns names: they are a pain.")
    print("""We will assign a new column name which is just a concatenation of summary statics
    name and pass band filter number.\n""")
    df_ts_agg.columns = ["{}_{}".format(col1, col2) for col1, col2 in df_ts_agg.columns]
    print("Column names: ", df_ts_agg.keys())

    print("""\n\nNow we merge the object meta data with the object time series summary values
    into a single data frame that we will use for training\n""")
    print(df_ts_agg.isnull().sum().sum())
    return df_ts


def merge_time_series_summary_and_metadata(df_ts_summary, df_metadata):
    df_merged = pd.merge(df_ts_summary, df_metadata, on='object_id', how='left')
    df_merged.drop(columns=['distmod', 'ra', 'decl', 'gal_l', 'gal_b','hostgal_specz'], inplace=True)
    df_merged.fillna(1.0)
    df_merged.replace([np.inf], 1.0)
    df_merged.replace([-np.inf], -1.0)
    return df_merged



class TimeSeriesReader:
    def __init__(self, fname='data/test_set.csv'):
        self.time_series_file = csv.reader(open(fname), delimiter=',')
        self.header_line = next(self.time_series_file)
        self.header_dtypes = [np.int, np.float, np.int, np.float, np.float, np.bool]
        print(self.header_line)
        self.line_buffer = []
        self.file_good = True


    def has_lines_to_read(self):
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
                if number_read < number or number==-1:
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
        if verbose:
            print("Number found: ", number_read)
        # Depending on the return format requested
        if return_format == "dict" or return_format == "pandas":
            result_dict = {}
            column_major = np.transpose(self.line_buffer)
            for i, (name, dtype) in enumerate(zip(self.header_line, self.header_dtypes)):
                if number_read != 0:
                    a = np.array(column_major[i], dtype=dtype)	
                else:
                    a = np.array([], dtype=dtype)
                result_dict[name] = a
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
            


