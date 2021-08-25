import os
import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn

from dcca_utils import DCCA_AM
from utils import concat_process
import logging

import pickle
# from notice_email import bug_notification_mail
import atexit
import scipy.io as sio

# loading data
eeg_dir = '../Features/China/eeg_used_4s/'
eye_dir = '../Features/China/eye_used/'
file_list = os.listdir(eeg_dir)
file_list.sort()

# design hyper-parameters
epochs = 100
eeg_input_dim = 310
eye_input_dim = 33
output_dim = 12
learning_rate = 1e-4
batch_size = 30

emotion_categories = 3
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

res_dir = './dcca_res/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

log_dir = './dcca_logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# preparing data
for f_id in file_list:
    print(f_id)
    logging.basicConfig(filename='./dcca_logs/output.log', level=logging.DEBUG)
    logging.debug('{}'.format(f_id))
    logging.debug('Task-Epoch-CCALoss-PredicLoss-PredicAcc')
    eeg_raw_data = np.load( os.path.join(eeg_dir, f_id) )
    eye_raw_data = sio.loadmat( os.path.join(eye_dir, f_id[:-4]+'.mat') )

    eeg_train_tmp, eeg_test_tmp, eye_train_tmp, eye_test_tmp, train_label, test_label = concat_process(
        eeg_raw_data, eye_raw_data)

    #train_eeg = preprocessing.scale(train_eeg)
    #train_eye = preprocessing.scale(train_eye)
    scaler = preprocessing.MinMaxScaler()
    train_eeg = scaler.fit_transform(eeg_train_tmp)
    train_eye = scaler.fit_transform(eye_train_tmp)
    test_eeg = scaler.fit_transform(eeg_test_tmp)
    test_eye = scaler.fit_transform(eye_test_tmp)

    sample_num = train_eeg.shape[0]
    batch_number = sample_num // batch_size

    train_eeg = torch.from_numpy(train_eeg).to(torch.float).to(device)
    train_eye = torch.from_numpy(train_eye).to(torch.float).to(device)
    test_eeg = torch.from_numpy(test_eeg).to(torch.float).to(device)
    test_eye = torch.from_numpy(test_eye).to(torch.float).to(device)
    train_label = torch.from_numpy(train_label).to(torch.long).to(device)
    test_label = torch.from_numpy(test_label).to(torch.long).to(device)

    # training process
    for hyper_choose in range(100):
        res_file_name = f_id[:-8] + '_' + str(hyper_choose)
        if res_file_name in os.listdir(res_dir):
            print(res_file_name)
            continue

        best_test_res = {}
        best_test_res['acc'] = 0
        best_test_res['predict_proba'] = None
        best_test_res['fused_feature'] = None
        best_test_res['transformed_eeg'] = None
        best_test_res['transformed_eye'] = None
        best_test_res['alpha'] = None
        best_test_res['true_label'] = None
        best_test_res['layer_size'] = None
        # try 100 combinations of different hidden units
        layer_sizes = [np.random.randint(100,200), np.random.randint(20,50), output_dim]
        logging.info('{}-{}'.format(layer_sizes[0], layer_sizes[1]))
        print(layer_sizes)
        mymodel = DCCA_AM(eeg_input_dim, eye_input_dim, layer_sizes, layer_sizes, output_dim, emotion_categories, device).to(device)
        optimizer_classifier = torch.optim.RMSprop(mymodel.parameters(), lr=learning_rate)
        optimizer_model1 = torch.optim.RMSprop(iter(list(mymodel.parameters())[0:8]), lr=learning_rate/2)
        optimizer_model2 = torch.optim.RMSprop(iter(list(mymodel.parameters())[8:16]), lr=learning_rate/2)
        class_loss_func = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            mymodel.train()
            best_acc = 0
            total_classification_loss = 0
            for b_id in range(batch_number+1):
                if b_id == batch_number:
                    train_eeg_used = train_eeg[batch_size*batch_number:, :]
                    train_eye_used = train_eye[batch_size*batch_number: , :]
                    train_label_used = train_label[batch_size*batch_number:]
                else:
                    train_eeg_used = train_eeg[b_id*batch_size:(b_id+1)*batch_size, :]
                    train_eye_used = train_eye[b_id*batch_size:(b_id+1)*batch_size, :]
                    train_label_used = train_label[b_id*batch_size:(b_id+1)*batch_size]
                try:
                    # predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, transformed_1, transformed_2, alpha  = mymodel(train_eeg_used, train_eye_used)
                    predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, alpha  = mymodel(train_eeg_used, train_eye_used)
                    predict_loss = class_loss_func(predict_out, train_label_used)

                    optimizer_model1.zero_grad()
                    optimizer_model2.zero_grad()
                    optimizer_classifier.zero_grad()

                    partial_h1 = torch.from_numpy(partial_h1).to(torch.float).to(device)
                    partial_h2 = torch.from_numpy(partial_h2).to(torch.float).to(device)

                    output1.backward(-0.1*partial_h1, retain_graph=True)
                    output2.backward(-0.1*partial_h2, retain_graph=True)
                    predict_loss.backward()

                    optimizer_model1.step()
                    optimizer_model2.step()
                    optimizer_classifier.step()
                except:
                    # bug_notification_mail('Local Bugs happend in: file_name: {} hyper_number: {} epoch_numner: {}.\nProgram Continues'.format(f_id, hyper_choose, epoch))
                    continue
            # for every epoch, evaluate the model on both train and test set
            mymodel.eval()
            predict_out_train, cca_loss_train, _, _, _, _, _, _  = mymodel(train_eeg, train_eye)
            predict_loss_train = class_loss_func(predict_out_train, train_label)
            accuracy_train = np.sum(np.argmax(predict_out_train.detach().cpu().numpy(), axis=1) == train_label.detach().cpu().numpy()) / predict_out_train.shape[0]

            predict_out_test, cca_loss_test, output_1_test, output_2_test, _, _, fused_tensor_test, attention_weight_test  = mymodel(test_eeg, test_eye)
            predict_loss_test = class_loss_func(predict_out_test, test_label)
            accuracy_test = np.sum(np.argmax(predict_out_test.detach().cpu().numpy(), axis=1) == test_label.detach().cpu().numpy()) / predict_out_test.shape[0]

            if accuracy_test > best_test_res['acc']:
                best_test_res['acc'] = accuracy_test
                best_test_res['layer_size'] = layer_sizes
                best_test_res['predict_proba'] = predict_out_test.detach().cpu().data
                best_test_res['fused_feature'] = fused_tensor_test
                best_test_res['transformed_eeg'] = output_1_test.detach().cpu().data
                best_test_res['transformed_eye'] = output_2_test.detach().cpu().data
                best_test_res['alpha'] = attention_weight_test
                best_test_res['true_label'] = test_label.detach().cpu().data

            print('Epoch: {} -- Train CCA loss is: {} -- Train loss: {} -- Train accuracy: {}'.format(epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
            print('Epoch: {} -- Test CCA loss is: {} -- Test loss: {} -- Test accuracy: {}'.format(epoch, cca_loss_test, predict_loss_test.data, accuracy_test))
            print('\n')
            logging.info('\tTrain\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
            logging.info('\tTest\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_test, predict_loss_test.data, accuracy_test))

        pickle.dump(best_test_res, open( os.path.join(res_dir, f_id[:-4]+'_'+str(hyper_choose)), 'wb'  ))

@atexit.register
def goodbye():
    bug_notification_mail('Local program ended successfully!')


