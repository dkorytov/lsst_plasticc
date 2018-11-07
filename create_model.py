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

from xgboost import XGBClassifier

import StringIO
import csv
import time
import gc

from summary_statistics import get_summary_statistic_function_dict, create_summary_statistics
from util import merge_time_series_summary_and_metadata, TimeSeriesReader, concatenate_dictionaries

check_classifier = True
        
def create_decision_forest(model_fname="models/decision_tree.pckl"):
    ts_reader = TimeSeriesReader(fname = "data/training_set.csv")
    metadata_df = pd.read_csv("data/training_set_metadata.csv")
    summary_functions = get_summary_statistic_function_dict()
    t1 = time.time()
    summary_dicts = []
    print("reading...")
    while(ts_reader.has_lines_to_read()):
        ts_dict = ts_reader.read_time_series_number(10, return_format='dict', verbose=False)
        summary_dicts.append(create_summary_statistics(ts_dict, summary_functions))
    summary_dict = concatenate_dictionaries(summary_dicts)
    t3 = time.time()
    print("\t ", t3-t1)
    summary_df = pd.DataFrame.from_dict(summary_dict)
    print("merging...")
    model_df = merge_time_series_summary_and_metadata(summary_df, metadata_df)
    t4 = time.time()
    print("\t ", t4-t3)
    print("training...")
    if(check_classifier):
        model_df, test = sklearn.model_selection.train_test_split(model_df)
        x_test = test.drop(columns=['target', 'object_id'])
        y_test_true = test['target']
    x_train = model_df.drop(columns=['target', 'object_id'])
    y_train = model_df['target']
    
    # clf_df =  sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)
    clf_df = XGBClassifier()
    clf_df.fit(x_train, y_train)
    if(check_classifier):
        y_test_pred = clf_df.predict(x_test)
        score = sklearn.metrics.accuracy_score(y_test_true, y_test_pred)
        print("\n\tAccuracy score: {}\n".format(score))

    t5 = time.time()
    print("\t {:.2f}".format(t5-t4))    
    pickle.dump(clf_df, open(model_fname, 'wb'))

    
if __name__ == "__main__":
    create_decision_forest()
            



