#!/usr/bin/env python2.7

from __future__ import print_function, division


import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import sklearn as skl

import sklearn.model_selection
import sklearn.tree
import sklearn.metrics

def max_above_mean(data):
    return np.max(data)-np.average(data)

def min_below_mean(data):
    return np.min(data)-np.average(data)

def calc_mean(data):
    return data.mean()

def frac_max_flux(data):
    avg = np.average(data)
    return np.max(data-avg)/np.sum(data-avg)

def frac_min_flux(data):
    avg = np.average(data)
    return np.min(data-avg)/np.sum(data-avg)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def get_processed_data(location):
    train_ts = pd.read_csv(location+'/training_set.csv')
    train_md = pd.read_csv(location+'/training_set_metadata.csv')

    print("\n\nDrop time series data we aren't using yet\n")    
    train_ts = train_ts.drop(['flux_err', 'detected', 'mjd'], axis = 1)
    print(train_ts.head(10))
    
    print("\n\nGroup the time series by object and passband.\nWe will use a single value per object and passband combination group\n")
    train_grouped = train_ts.groupby(['object_id', 'passband'])


    print("\n\nFor each group (id + passband), we will calculate the mean, the maximum/minimum flux above the mean, and standard deviation of the time series\n")
    train_ts_agg = train_grouped.agg([np.mean, max_above_mean, min_below_mean, frac_max_flux, frac_min_flux]).reset_index()
    print("column names:", train_ts_agg.keys())

    print("\n\nThe column names now have a hierarchical structure with two levels. \nLets get rid of it\n") 
    train_ts_agg.columns = ([col1 if col1 != "flux" else col2 for col1, col2 in train_ts_agg.columns])
    print("column names: ", train_ts_agg.keys())

    print("\n\nLets use pivot to create a single row per object, with multiple entries per band and time serties property\n")
    train_ts_agg = train_ts_agg.pivot(index='object_id', columns='passband', values=['mean', 'max_above_mean', 'min_below_mean'])
    print("Column names: ", train_ts_agg.keys())
    
    print("\n\nAgain, lets get rid of the hierarchical columns names: they are a pain.")
    print("""We will assign a new column name which is just a concatenation of summary statics
    name and pass band filter number.\n""")
    train_ts_agg.columns = ["{}_{}".format(col1, col2) for col1, col2 in train_ts_agg.columns]
    print("Column names: ", train_ts_agg.keys())

    print("""\n\nNow we merge the object meta data with the object time series summary values
    into a single data frame that we will use for training\n""")
    print(train_ts_agg.isnull().sum().sum())
    given_data = pd.merge(train_md, train_ts_agg, on='object_id')
    given_data = given_data.drop(columns=['distmod', 'ra', 'decl', 'gal_l', 'gal_b',])
    return given_data



def simple_train(location = 'data'):
    data = get_processed_data(location=location)
    target_names = np.sort(data['target'].unique())
    print(data.isnull().sum().sum())
    print("\n\n\n")
    train, test = sklearn.model_selection.train_test_split(data)
    x_train = train.drop(columns=['target'])
    y_train = train['target']
    x_test = test.drop(columns=['target'])
    y_test_true = test['target']
    best_score = 0.0
    best_depth = 0
    for i in range(1, 10):
        cldf = sklearn.tree.DecisionTreeClassifier(max_depth=i)
        cldf.fit(x_train, y_train)
        y_test_pred = cldf.predict(x_test)
        score = sklearn.metrics.accuracy_score(y_test_true, y_test_pred)
        if score > best_score:
            best_score = score
            best_depth = i
        print("Depth: {}\n\tAccuracy score: {}\n".format(i,score))
        #print("Multiclass loss: ", sklearn.metrics.log_loss(y_test_true, y_test_pred, labels=a))
    cldf = sklearn.tree.DecisionTreeClassifier(max_depth=i)
    cldf.fit(x_train, y_train)
    y_test_pred = cldf.predict(x_test)
    cnfs_mtx = sklearn.metrics.confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)
    sklearn.tree.export_graphviz(cldf, out_file='tree.dot', feature_names=x_test.keys(),filled=True,rounded=True)

    plt.show()
    
if __name__ == "__main__":
    simple_train()

