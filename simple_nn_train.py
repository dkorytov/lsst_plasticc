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
    #train_ts = train_ts.drop(['flux_err', 'detected', 'mjd'], axis = 1)
    print(train_ts.head(10))
    
    print("\n\nGroup the time series by object and passband.\nWe will use a single value per object and passband combination group\n")
    train_grouped = train_ts.groupby(['object_id', 'passband'])

    ltime_seen = lambda g: time_seen(train_ts, g.index)
    ldf_dt_min = lambda g: df_dt_min(g, train_ts, g.index)
    ldf_dt_max = lambda g: df_dt_max(g, train_ts, g.index)
    ldf_dt_mean= lambda g: df_dt_mean(g, train_ts, g.index)
        
    print("\n\nFor each group (id + passband), we will calculate the mean, the maximum/minimum flux above the mean, and standard deviation of the time series\n")
    #train_ts_agg = train_grouped.agg({'flux':{'time_seen': ltime_seen, 'df_dt_min': ldf_dt_min, 'df_dt_max': ldf_dt_max, 'df_dt_mean': ldf_dt_mean, 'mean': np.mean, 'max_above': max_above_mean, 'min_below': min_below_mean, 'frac_max': frac_max_flux, 'frac_min': frac_min_flux }, 'detected':{'detected_frac': np.average}}).reset_index()
    train_ts_agg = train_grouped.agg({'flux':{'mean': np.mean, 'max_above': max_above_mean, 'min_below': min_below_mean, 'frac_max': frac_max_flux, 'frac_min': frac_min_flux }, 'detected':{'detected_frac': np.average}}).reset_index()

    print("column names:", train_ts_agg.keys())

    print("\n\nThe column names now have a hierarchical structure with two levels. \nLets get rid of it\n") 
    train_ts_agg.columns = ([col1 if col1 != "flux" else col2 for col1, col2 in train_ts_agg.columns])
    print("column names: ", train_ts_agg.keys())

    print("\n\nLets use pivot to create a single row per object, with multiple entries per band and time serties property\n")
    train_ts_agg = train_ts_agg.pivot(index='object_id', columns='passband')#, values=['mean', 'max_above_mean', 'min_below_mean', 'frac_max_flux', 'frac_min_flux', 'ltime_seen'])
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
    given_data = given_data.drop(columns=['distmod', 'ra', 'decl', 'gal_l', 'gal_b'])
    given_data.fillna(1.0)
#    print("nans: ", given_data.isnan().sum().sum())
    given_data.replace([np.inf], 1.0)
    given_data.replace([-np.inf], -1.0)

    return given_data

def rescale_by_column(data):
    # col_mean = np.mean(data, axis=1)
    col_min = np.min(data, axis=1)
    data -= col_min[:,None]
    col_max = np.max(data, axis=1)
    data /= col_max[:,None]
    return data

def get_train_test_data(data):
    train, test = sklearn.model_selection.train_test_split(data)
    x_train = np.array(train.drop(columns=['target', 'object_id', 'hostgal_specz']))
    y_train = train['target']
    x_test = np.array(test.drop(columns=['target', 'object_id', 'hostgal_specz']))
    y_test_true = test['target']


    ## Rescaling the data by column
    rescaled_data = rescale_by_column(np.vstack([x_train, x_test]))
    x_train = rescaled_data[: np.shape(y_train)[0],:]
    x_test = rescaled_data[np.shape(y_train)[0]:,:]

    return x_train, y_train, x_test, y_test_true

def simple_train(location = 'data'):
    data = get_processed_data(location=location)
    x_train, y_train, x_test, y_test_true = get_train_test_data(data)
    target_names = np.sort(data['target'].unique())
    best_score = 0.0
    best_depth = 0
    for i in range(1, 1):
        cldf = sklearn.tree.DecisionTreeClassifier(max_depth=i)
        cldf.fit(x_train, y_train)
        y_test_pred = cldf.predict(x_test)
        score = sklearn.metrics.accuracy_score(y_test_true, y_test_pred)
        if score > best_score:
            best_score = score
            best_depth = i
        print("Depth: {}\n\tAccuracy score: {}\n".format(i,score))
        #print("Multiclass loss: ", sklearn.metrics.log_loss(y_test_true, y_test_pred, labels=a))
    cldf = sklearn.tree.DecisionTreeClassifier(max_depth=6)
    cldf.fit(x_train, y_train)
    y_test_pred = cldf.predict(x_test)
    score = sklearn.metrics.accuracy_score(y_test_true, y_test_pred)
    print("Depth: {}\n\tAccuracy score: {}\n".format(i,score))
    cnfs_mtx = sklearn.metrics.confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)
    pickle.dump(cldf, open("models/decision_tree.pckl", 'wb'))
    sklearn.tree.export_graphviz(cldf, out_file='tree.dot', feature_names=x_test.keys(),filled=True,rounded=True)
    plt.show()

def random_forest_train(location = 'data'):
    data = get_processed_data(location=location)
    x_train, y_train, x_test, y_test_true = get_train_test_data(data)
    target_names = np.sort(data['target'].unique())
    forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)
    #forest = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1000, random_state=0)
    print("Training...")
    forest.fit(x_train, y_train)
    print("Done training...")
    y_test_pred = forest.predict(x_test)
    y_train_pred = forest.predict(x_train)
    score = sklearn.metrics.accuracy_score(y_test_true, y_test_pred)
    
    cnfs_mtx = sklearn.metrics.confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    columns = x_train.columns
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(x_train.shape[1]):
        print("{}. feature {} ({})".format(f + 1, columns[indices[f]], importances[indices[f]]))
        
    print("\n\tAccuracy score: {}\n".format(score))

    # Plot the feature importances of the forest
    f = plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), columns[indices], rotation='vertical')
    plt.xlim([-1, x_train.shape[1]])
    plt.tight_layout()

    # retries = [42, 90]
    # for retry in retries:
    #     slct = y_train_pred == retry
    #     x_train_retry, y_train_retry = x_test[slct], y_test[slct]

    plt.show()

def relabel(labels, data):
    result = np.copy(data)
    for i, label in enumerate(labels):
        slct = data==label
        result[slct] = i
    return result

def ann_train(location='data'):
    # data = get_processed_data(location=location)
    # print(data.keys())
    # data.to_csv("data/data.csv")
    data = pd.read_csv("data/data.csv").drop(columns=['Unnamed: 0'])
    print(data.keys())
    # exit()
    x_train, y_train, x_test, y_test_true = get_train_test_data(data)

    # x_train = np.array(x_train)
    # x_test = np.array(x_test)

    # x_train, x_test = rescale_by_column( np.vstack([x_train, x_test]) )



    target_names = np.sort(data['target'].unique())

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    import keras.optimizers
    from keras import backend as K
    batch_size = 4
    num_classes = len(target_names)
    epochs = 200
    
    y_train = keras.utils.to_categorical(relabel(target_names, y_train), num_classes)
    y_test = keras.utils.to_categorical(relabel(target_names, y_test_true), num_classes)
    
    input_shape = np.shape(x_train)
    print(input_shape)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    learning_rate = 5.0
    decay_rate = 1.0
    adaDelta = keras.optimizers.Adadelta(lr=learning_rate, decay=decay_rate)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  #optimizer=adaDelta,
                  metrics=['accuracy'])
    
    ModelFit = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    train_acc= ModelFit.history['acc']
    val_acc= ModelFit.history['val_acc']
    epoch_array = range(1, epochs+1)


    fig, ax = plt.subplots(2,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax[0].plot(epoch_array,train_loss)
    ax[0].plot(epoch_array,val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss','val_loss'])


    ax[1].plot(epoch_array,train_acc)
    ax[1].plot(epoch_array,val_acc)
    ax[1].set_ylabel('acc')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[1].legend(['train_acc','val_acc'])

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    epoch_array = range(1, epochs+1)


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epoch_array,train_loss)
    ax.plot(epoch_array,val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss','val_loss'])

    plt.show()



    plt.figure(322)
    y_test_pred = model.predict_classes(x_test)
    cnfs_mtx = sklearn.metrics.confusion_matrix(relabel(target_names, y_test_true), y_test_pred)
    plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)
    plt.show()




def rnn_train(location='data'):
    # data = get_processed_data(location=location)
    # print(data.keys())
    # data.to_csv("data/data.csv")
    data = pd.read_csv("data/data.csv").drop(columns=['Unnamed: 0'])
    print(data.keys())
    # exit()
    x_train, y_train, x_test, y_test_true = get_train_test_data(data)

    # x_train = np.array(x_train)
    # x_test = np.array(x_test)

    # x_train, x_test = rescale_by_column( np.vstack([x_train, x_test]) )



    target_names = np.sort(data['target'].unique())

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, LSTM
    from keras.layers import Conv2D, MaxPooling2D
    import keras.optimizers
    from keras import backend as K
    batch_size = 4
    num_classes = len(target_names)
    epochs = 200

    y_train = keras.utils.to_categorical(relabel(target_names, y_train), num_classes)
    y_test = keras.utils.to_categorical(relabel(target_names, y_test_true), num_classes)

    input_shape = np.shape(x_train)
    print(input_shape)


    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(input_shape[1], )))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    rmsprop = keras.optimizers.RMSprop(lr=learning_rate, decay=decay_rate)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    print(x_train.shape, y_train.shape)

    ModelFit = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=2,
                         validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    train_acc = ModelFit.history['acc']
    val_acc = ModelFit.history['val_acc']
    epoch_array = range(1, epochs + 1)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
    ax[0].plot(epoch_array, train_loss)
    ax[0].plot(epoch_array, val_loss)
    ax[0].set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[0].legend(['train_loss', 'val_loss'])

    ax[1].plot(epoch_array, train_acc)
    ax[1].plot(epoch_array, val_acc)
    ax[1].set_ylabel('acc')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax[1].legend(['train_acc', 'val_acc'])

    train_loss = ModelFit.history['loss']
    val_loss = ModelFit.history['val_loss']
    epoch_array = range(1, epochs + 1)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(7, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.02)
    ax.plot(epoch_array, train_loss)
    ax.plot(epoch_array, val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss', 'val_loss'])

    plt.show()

    plt.figure(322)
    y_test_pred = model.predict_classes(x_test)
    cnfs_mtx = sklearn.metrics.confusion_matrix(relabel(target_names, y_test_true), y_test_pred)
    plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)
    plt.show()





def feature_selection(location='data'):
    data = get_processed_data(location=location)
    x_train, y_train, x_test, y_test_true = get_train_test_data(data)
    target_names = np.sort(data['target'].unique())
    clf_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    rfecv = sklearn.feature_selection.RFECV(estimator=clf_rf, step=1, cv=5, scoring='accuracy')
    rfecv = rfecv.fit(x_train, y_train)
    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', x_train.columns[rfecv.support_])
    
if __name__ == "__main__":
    # simple_train()
    # random_forest_train()
    # feature_selection()
    # ann_train()
    rnn_train()