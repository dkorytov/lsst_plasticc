
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
    

def get_summary_statistic_function_dict():
    """ This method returns a dict of functions that compute 
    summary statistics

    :return: a python dict containing time series summary functions 
    """
    return_dict = {}
    bands = [0,1,2,3,4,5]
    band_functions = {}
    band_functions['max'] = band_max
    band_functions['min'] = band_min
    band_functions['mean'] = band_mean
    band_functions['median'] = band_median
    band_functions['max'] = band_mean
    for band in bands:
        for band_funct_name in band_functions.keys():
            funct = band_functions[band_funct_name]
            band_funct = lambda dict_ts: funct(dict_ts, band)
            return_dict["{}_{}".format(band_funct_name, band)]= band_funct
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
        output_dict[func_name].append(functions[func_name](dict_time_series))
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
    
