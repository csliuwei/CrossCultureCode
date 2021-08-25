import os
import numpy as np
import scipy.io as sio
from keras import backend as K
from keras.layers import Dense, Activation, Input, merge, Lambda
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
# from liblinearsvm import liblinearclassify
import pickle

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from utils import concat_process

def get_session(gpu_fraction=0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

eegDir = '../Features/China/eeg_used_4s/'
eyeDir = '../Features/China/eye_used/'
fileNames = os.listdir(eegDir)
fileNames.sort()

eyeNames = os.listdir(eyeDir)
eyeNames.sort()

# loading all data into memory
eeg_data_list = []
eye_data_list = []

bdae_transformed_dir = './bdae_transformed/'
if not os.path.exists(bdae_transformed_dir):
    os.mkdir(bdae_transformed_dir)

for it in [35]:
    eeg_raw_data = np.load( os.path.join(eegDir, fileNames[it]))
    eye_raw_data = sio.loadmat(os.path.join(eyeDir, eyeNames[it]))

    eeg_train_tmp, eeg_test_tmp, eye_train_tmp, eye_test_tmp,  train_label, test_label = concat_process(eeg_raw_data, eye_raw_data)
    data1 = np.vstack((eeg_train_tmp, eeg_test_tmp))
    data2 = np.vstack((eye_train_tmp, eye_test_tmp))

    scaler = MinMaxScaler()
    # normalize
    eeg_train = scaler.fit_transform(data1)
    eye_train = scaler.fit_transform(data2)

    # network parameters
    x_row, x_col = eeg_train.shape
    y_row, y_col = eye_train.shape
    del scaler

    epochs = [1000,700, 500,300, 200,100, 50, 30, 10]
    hidden_units = [700,500,200,170,150,130,110,90,70,50]
    el = len(epochs)
    hl = len(hidden_units)

    for ei, this_epoch in enumerate(epochs):
        for hi, this_hidden in enumerate(hidden_units):
            res_row = ei * hl + hi
            # res_col = it
            # RBM initialize
            rbm1 = BernoulliRBM(n_components=this_hidden, batch_size=100,n_iter=20)
            rbm2 = BernoulliRBM(n_components=this_hidden,batch_size=100, n_iter=20)
            rbm3 = BernoulliRBM(n_components=this_hidden,batch_size=100, n_iter=20)
            # rbm1
            print('RBM 1 training\n')
            rbm1.fit(eeg_train)
            hidden_eeg = rbm1.transform(eeg_train)
            weights_eeg = rbm1.components_
            # rbm2
            print('RBM 2 training\n')
            rbm2.fit(eye_train)
            hidden_eye = rbm2.transform(eye_train)
            weights_eye = rbm2.components_
            # rbm3
            print('RBM 3 training \n')
            conca_data = np.append(hidden_eeg, hidden_eye, axis=1)
            rbm3.fit(conca_data)
            # hidden_merge = rbm3.transform(conca_data)
            weights_merge = rbm3.components_

            # netwrok structure
            def get_eeg_part(nparray):
                global this_hidden
                # row, col = nparray.get_shape()
                return nparray[:,:this_hidden]
            def get_eye_part(nparray):
                global this_hidden
                # row, col = nparray.get_shape()
                return nparray[:, this_hidden:]

            print('Model structure begin ... \n')
            print('Input layers ...\n')
            x_input = Input(shape=(x_col,), name='x_input')
            y_input = Input(shape=(y_col,), name='y_input')

            print('hidden layers ...\n')
            x_hidden = Dense(this_hidden, weights=[weights_eeg.T, rbm1.intercept_hidden_], activation='sigmoid',name='x_hidden')(x_input)
            y_hidden = Dense(this_hidden, weights=[weights_eye.T, rbm2.intercept_hidden_], activation='sigmoid',name='y_hidden')(y_input)

            print('merge layers ... \n')
            merge_xy = merge([x_hidden, y_hidden], mode='concat')
            feature_layer = Dense(this_hidden, weights=[weights_merge.T, rbm3.intercept_hidden_], activation='sigmoid', name='merged')(merge_xy)
            # decoding
            print('decoding processing \n')
            merge_xy_t = Dense(2*this_hidden,weights=[weights_merge, rbm3.intercept_visible_], activation='sigmoid',name='merge_t')(feature_layer)
            x_hidden_t = Lambda(get_eeg_part, output_shape=(this_hidden,))(merge_xy_t)
            y_hidden_t = Lambda(get_eye_part, output_shape=(this_hidden,))(merge_xy_t)

            x_recon = Dense(x_col, weights=[weights_eeg, rbm1.intercept_visible_], activation='sigmoid',name='x_recon')(x_hidden_t)
            y_recon = Dense(y_col, weights=[weights_eye, rbm2.intercept_visible_], activation='sigmoid',name='y_recon')(y_hidden_t)

            model = Model(input=[x_input, y_input], output=[x_recon, y_recon])
            model.compile(optimizer='rmsprop',loss='mean_squared_error')
            model.fit([eeg_train, eye_train],[eeg_train, eye_train], nb_epoch=this_epoch, batch_size=100)
            # print('\n')
            print('\n extracting middle features\n')

            feature_res = K.function([model.layers[0].input, model.layers[1].input],[model.layers[5].output])

            ## get the extracted feature
            train_features = feature_res([eeg_train, eye_train])[0]

            f1 = open(bdae_transformed_dir + fileNames[it][:-4] +'_'+str(res_row)+'_'+str(this_epoch)+'_'+str(this_hidden), 'wb')
            pickle.dump(train_features, f1)
            f1.close()

