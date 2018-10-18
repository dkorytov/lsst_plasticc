import numpy as np
import matplotlib.pylab as plt
import pandas as pd


training_set = pd.read_csv("data/training_set.csv")
print('training set', training_set.keys())

meta_data = pd.read_csv("data/training_set_metadata.csv")

print('metadata', meta_data.keys())

types = np.sort(np.unique(meta_data['target']))

print(np.shape(types))
#
# for obj_type in types:
#
#     print(obj_type)
#     object_ids = (meta_data[meta_data['target'] == obj_type]['object_id']).sample(9)
#


###########################################

PlotGal = False

if PlotGal == True:

    targets = np.hstack([np.unique(meta_data['target']), [99]])
    target_map = {j:i for i, j in enumerate(targets)}
    target_ids = [target_map[i] for i in meta_data['target']]
    meta_data['target_id'] = target_ids

    galactic_cut = meta_data['hostgal_specz'] == 0
    plt.figure(figsize=(10, 8))
    plt.hist(meta_data[galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Galactic')
    plt.hist(meta_data[~galactic_cut]['target_id'], 15, (0, 15), alpha=0.5, label='Extragalactic')
    plt.xticks(np.arange(15)+0.5, targets)
    plt.gca().set_yscale("log")
    plt.xlabel('Class')
    plt.ylabel('Counts')
    plt.xlim(0, 15)
    plt.legend()


###########################################



PlotTimeSeries = False

if PlotTimeSeries == True:
    class_ids = meta_data['target'].unique()
    class_ids.sort(axis=0)


    oids = [(c, meta_data[meta_data['target']==c].sample(n=3, random_state=2018)['object_id'].values) for c in class_ids]
    oids

    for cid, oid in oids:
        plt.figure(figsize=(10, 10))
        for band in range(6):
            plt.subplot(6, 1, band + 1)
            if band == 0:
                plt.title('Target {} (object_id = {})'.format(cid, oid))
            plt.ylabel('Band {}'.format(band))
            for i in range(3):
                ts = training_set[((training_set['object_id'] == oid[i]) & (training_set['passband'] == band))]
                plt.plot(ts['mjd'], ts['flux'], 'x-')
        plt.show()

##################################################################

PlotTimeSeries2 = False

if PlotTimeSeries2 == True:

    class_ids = meta_data['target'].unique()
    class_ids.sort(axis=0)


    oids = [(c, meta_data[meta_data['target']==c].sample(n=3, random_state=2018)['object_id'].values) for c in class_ids]

    # print('objects in classes')
    # print(*oids, sep= '\n')

    for cid, oid in oids:

        print(cid, oid)



        plt.figure(figsize=(10, 10))
        for cls in range(class_ids.shape[0]):
            plt.subplot(3, 1, cls + 1)
            # if band == 0:
            #     plt.title('Target {} (object_id = {})'.format(cid, oid))
            # plt.ylabel('Band {}'.format(band))
            for i in range(3):
                ts = training_set[ ((training_set['object_id'] == oid[i]) & (training_set['passband'] == band))]


                plt.plot(ts['mjd'], ts['flux'], 'x-')
        plt.show()

###################################

#### Naive classification scheme - CREATE train/validation set ####

###################################
class_ids = meta_data['target'].unique()
class_ids.sort(axis=0)

num_bands = 6
num_objects = 100
obj_summary_shape = 9
num_ip = num_bands*obj_summary_shape ## num_bands * [total, min, max, total{dx/dt}, min{dx/dt},
# max{dx/dt},
# total{dx/dt}]
num_classes = class_ids.shape[0]

meta_shape = 11

# TrainingData = np.zeros(shape=(num_objects*num_classes, num_ip + 1 + meta_shape))

tot_train = meta_data.shape[0]
TrainingData = np.zeros(shape=(tot_train, num_ip + 1 + meta_shape))

# TrainingData = np.zeros(shape=num_ip + 1 + meta_shape)


max_tsteps = 0
t_min = 60000
t_max = 60000


obj_id = np.unique(training_set['object_id'])[0]


class_id_norm = 0

idx = 0


GenerateTrainingSummary = False

if GenerateTrainingSummary:

    for class_id in class_ids:

        print('============================', 'Class ID', class_id)




        object_ids = meta_data[meta_data['target'] == class_id]['object_id']  ## All objects
        # object_ids = meta_data[meta_data['target'] == obj_type]['object_id'].sample(num_objects)

        num_obj_per_class = object_ids.shape[0]



        obj_id_norm = 0
        for obj_id in object_ids:
            # print('==========', 'Object ID', obj_id)


            obj_summary = []

            for band in range(num_bands):
                # print band

                # band_summary = []

                ts_selected = training_set[ (training_set['object_id'] == obj_id) & (training_set['passband'] == band)]
                x_train_i = ts_selected['flux']
                xerr_train_i = ts_selected['flux_err']

                t_train_i = ts_selected['mjd']
                y_train_i = class_id

                # print('band', band, 'flux shape', x_train_i.shape, 'mjd', t_train_i.shape)

                plt.figure(342+class_id)
                plt.errorbar(t_train_i, x_train_i, yerr= xerr_train_i, fmt='x', alpha = 0.8)
                # plt.errorbar(np.arange(t_train_i.shape[0]), x_train_i, yerr=xerr_train_i, fmt='x', alpha=0.8)
                # plt.plot(np.arange(t_train_i.shape[0]), t_train_i, alpha=0.8)


                # print t_train_i.min(), t_train_i.max()

                max_tsteps = np.max( [max_tsteps, np.max(t_train_i.shape[0])])
                t_min = np.min( [t_min, t_train_i.values[0] ])
                t_max = np.max( [t_max, t_train_i.values[-1]])


                ####
                dxdt = np.gradient(x_train_i, t_train_i)

                flux_sum = x_train_i.sum()



                obj_band_summary = np.array(  [ class_id_norm, np.mean(x_train_i), np.std(x_train_i),
                                                np.max(x_train_i),
                                           np.min(x_train_i), np.mean(dxdt), np.std(dxdt), np.max(dxdt),
                                           np.min(dxdt)  ] )


                # TrainingData[class_id_norm*num_classes + obj_id_norm*num_objects: class_id_norm*num_classes +
                #                                     num_objects*obj_id_norm + 1,
                # num_bands*band: num_bands*band+ obj_summary_shape]= obj_summary

                # print(class_id_norm*num_classes + obj_id_norm*num_objects)
                # print(class_id_norm*num_classes)

                # TrainingData = np.hstack( [obj_band_summary])

                obj_summary = np.hstack([obj_summary, obj_band_summary])





            meta_obj_data = meta_data[meta_data['object_id'] == obj_id].values

            obj_summary = np.hstack( [class_id_norm, obj_summary,  meta_obj_data[:,1:][0]   ])


            obj_id_norm += 1
            idx += 1

            TrainingData[idx - 1:, ] = obj_summary
            print idx
            # TrainingData = np.vstack([obj_summary, TrainingData])

        class_id_norm += 1
        print('=========================================clid', class_id_norm, obj_id_norm)




    TrainingData = TrainingData[1:, :]

    # print max_tsteps
    # print t_min
    # print t_max

    print('====================================idx',idx)

    np.savetxt('data/TrainingSummaryData.txt', TrainingData)

else:
    TrainingData = np.loadtxt('data/TrainingSummaryData.txt')
# x_train = training_set['passband']
# y_train = meta_data['target']



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 4
num_classes = 14
epochs = 10
learning_rate = 0.1
decay_rate = 0.0


x_train = TrainingData[:, 1:]
y_train = TrainingData[:, 0]


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(65,)))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

adaDelta = keras.optimizers.Adadelta(lr=learning_rate, decay=decay_rate)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adaDelta,
              metrics=['accuracy'])

ModelFit = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split= 0.2)



score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



plotLossAcc = False
if plotLossAcc:
    import matplotlib.pylab as plt

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

    plt.show()


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

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
#
# import sklearn as skl
#
# score = skl.metrics.accuracy_score(y_test_true, y_test_pred)
# print("Depth: {}\n\tAccuracy score: {}\n".format(i,score))
# cnfs_mtx = skl.metrics.confusion_matrix(y_test_true, y_test_pred)
# plot_confusion_matrix(cnfs_mtx, target_names, normalize=True)
# pickle.dump(cldf, open("models/decision_tree.pckl", 'wb'))
# sklearn.tree.export_graphviz(cldf, out_file='tree.dot', feature_names=x_test.keys(),filled=True,rounded=True)
# plt.show()