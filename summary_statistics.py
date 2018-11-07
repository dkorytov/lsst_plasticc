
from __future__ import print_function, division


import itertools
import pickle
import numpy as np
import pandas as pd
import csv


def band_mean(dict_ts, band):
    slct = dict_ts['passband'] == band
    return np.mean(dict_ts['flux'][slct])
    
def band_max(dict_ts, band):
    slct = dict_ts['passband'] == band
    return np.max(dict_ts['flux'][slct])

def band_min(dict_ts, band):
    slct = dict_ts['passband'] == band
    return np.min(dict_ts['flux'][slct])

def band_median(dict_ts, band):
    slct = dict_ts['passband'] == band
    return np.median(dict_ts['flux'][slct])

def band_max_median(dict_ts, band):
    slct = dict_ts['passband'] == band
    median = np.median(dict_ts['flux'][slct])
    max_v = np.max(dict_ts['flux'][slct])
    return max_v-median

def band_min_median(dict_ts, band):
    slct = dict_ts['passband'] == band
    median = np.median(dict_ts['flux'][slct])
    min_v = np.min(dict_ts['flux'][slct])
    return median-min_v

def band_time_seen(dict_ts, band):
    slct1 = dict_ts['passband'] == band
    slct2 = dict_ts['detected'] == 1
    time = dict_ts['mjd'][slct1 & slct2]
    if time.size ==0:
        return np.max(dict_ts['mjd']) - np.min(dict_ts['mjd'])
    else:
        return np.max(time) - np.min(time)

def band_std(dict_ts, band):
    slct = dict_ts['passband'] == band
    return np.std(dict_ts['flux'])

def frac_detected(dict_ts):
    return np.sum(dict_ts['detected'])/dict_ts['detected'].size

def flux_std(dict_ts):
    return np.std(dict_ts['flux'])

def flux_max(dict_ts):
    return np.max(dict_ts['flux'])

def flux_min(dict_ts):
    return np.min(dict_ts['flux'])

def flux_mean(dict_ts):
    return np.mean(dict_ts['flux'])

def flux_median(dict_ts):
    return np.median(dict_ts['flux'])

def flux_skew(dict_ts):
    return np.median(dict_ts['flux']) - np.mean(dict_ts['flux'])

def time_seen(dict_ts):
    slct = dict_ts['detected'] == 1
    time = dict_ts['mjd'][slct]
    if time.size ==0:
        return np.max(dict_ts['mjd']) - np.min(dict_ts['mjd'])
    else:
        return np.max(time) - np.min(time)


def get_summary_statistic_function_dict():
    """ This method returns a dict of functions that compute 
    summary statistics

    :return: a python dict containing time series summary functions 
    """
    return_dict = {}
    # bands = [0,1,2,3,4,5]
    # band_functions = {'min': band_min,
    #                   'max': band_max,
    #                   'median': band_median,
    #                   'max_med': band_max_median,
    #                   'min_med': band_min_median,
    #                   'std': band_std,
    #                   'timeseen': band_time_seen,}
    # for band in bands:
    #     for band_funct_name in band_functions.keys():
    #         funct = band_functions[band_funct_name]
    #         return_dict["{}_{}".format(band_funct_name, band)]= lambda dict_ts, funct=funct, band=band: funct(dict_ts, band)
    # return_dict['frac_detected']=frac_detected
    return_dict = {'std':  flux_std,
                   'max':  flux_max,
                   'min':  flux_min,
                   'mean': flux_mean,
                   'skew': flux_skew,
                   'detect': frac_detected,
                   'time_seen': time_seen,
                   }
    return return_dict

    


def create_summary_statistics(dict_time_series, functions):
    """ This method accepts a dict containing the time series data

    :param dict_time_series: python dict of time series columns
    :param functions: python dict of functions
    :return: a python dict containing columns of object_id and summary statistics
    """
    output_dict = {}
    output_dict['object_id'] = []
    for func_name in functions.keys():
        output_dict[func_name] = []
    object_ids = np.unique(dict_time_series['object_id'])
    for object_id in object_ids:
        dict_time_series_obj = select_obj_id(dict_time_series, object_id)
        create_summary_statistics_individual(dict_time_series_obj, functions, output_dict)
        output_dict["object_id"].append(object_id)
    return output_dict

def create_summary_statistics_individual(dict_time_series, functions, output_dict):
    """This method accepts a dict with the time series data for one object and function dict
    that contains summary statistics

    :param dict_time_series: a python dict with a key for each data column in csv file
    :param functtions: a python dict with each element being a function that accepts the dict_time_series
    and returns a single value summary
    :param output_dict: 
    :return: a list of summary statistics in the order that they were given

    """
    for func_name in functions.keys():
        val = functions[func_name](dict_time_series)
        output_dict[func_name].append(val)
    return

def select_obj_id(dict_time_series, obj_id):
    """ This method extracts a single object from the time series data
    :param dict_time_series: a python dict containing the time sereis data columns
    :param obj_id: the id number of the number that we want to extract
    :return: a python dict containing only the time series data for the object specified
    """
    result = {}
    slct = dict_time_series['object_id'] == obj_id
    for key in dict_time_series.keys():
        result[key] = dict_time_series[key][slct]
    return result
    
