import numpy as np
import os
import torch
import torch.nn as nn 

from sklearn import preprocessing 
import pickle 
import scipy.io as sio 
from utils import generating_data 
import time 


class dnn(nn.Module):
    def __init__(self, in_num=310, h1=128, h2=64, h3=32, out_num=3):
        super(dnn, self).__init__()
        self.dnn_net = nn.Sequential(
            nn.Linear(in_num, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_num)
        )
    
    def forward(self, x):
        return self.dnn_net(x)

def train_dnn(train_data, test_data, train_label, test_label, lr, device):
    model = dnn(310, 128, 64, 32, 3)
    model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    train_label = train_label.to(device)
    test_label = test_label.to(device)

    epoch_num = 15000
    learning_rate = lr
    batch_size= 50

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    myloss = nn.CrossEntropyLoss()

    best_test_res = {
        'acc':0,
        'predict_label':None,
        'trur_label':None
    }

    for ep in range(epoch_num):
        model.train()
        output = model(train_data)
        loss = myloss(output, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(torch.argmax(output, 1) == train_label) / train_data.shape[0]

        print('Epoch : {} -- TrainLoss : {} -- TrainAcc : {}\n'.format(ep, loss.cpu().data, train_acc.cpu().data))

        model.eval()
        test_output = model(test_data)
        test_loss = myloss(test_data, test_label)
        test_acc = torch.sum(torch.argmax(test_output, 1) == test_label) / test_data.shape[0]
        print('Epoch : {} -- TestLoss : {} -- TestAcc : {}\n'.format(ep, test_loss.cpu().data, test_acc.cpu().data))

        # save the best results    
        if best_test_res['acc'] < test_acc.cpu().data:
            best_test_res['acc'] = test_acc.cpu().data 
            best_test_res['predict_label'] = torch.argmax(test_output, 0).cpu().numpy()
            best_test_res['true_label'] = test_label.cpu().numpy()
            print('update res')
    return best_test_res      

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

eeg_dir = '../Features/China/feature_smooth_eeg_1s/'
eeg_file_list = os.listdir(eeg_dir)
eeg_file_list.sort()

res_dir = './11_dnn_eeg_1s_baseline/'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

feature_list = ['de_LDS']
label_list = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]) + 1 # -1 for 
learning_rate = [0.00001, 0.00003, 0.00005, 0.00007, 0.00009,0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09]

for item in eeg_file_list:
    print(item)
    all_data = sio.loadmat(os.path.join(eeg_dir, item))
    
    for fn in feature_list:
        print(fn)
        train_data, test_data, train_label, test_label = generating_data(all_data, label_list, fn)
        train_data = preprocessing.scale(train_data)
        test_data = preprocessing.scale(test_data)

        train_data_tensor = torch.from_numpy(train_data).to(torch.float)
        test_data_tensor = torch.from_numpy(test_data).to(torch.float)
        train_label_tensor = torch.from_numpy(train_label).to(torch.long)
        test_label_tensor = torch.from_numpy(test_label).to(torch.long)

        for idx, lr in enumerate(learning_rate):
            best_res = train_dnn(train_data_tensor, test_data_tensor, train_label_tensor, test_label_tensor, lr, device)
            if not os.path.exists(os.path.join(res_dir, fn+str(idx))):
                os.mkdir(os.path.join(res_dir, fn+str(idx)))
            pickle.dump(best_res, open(os.path.join(res_dir, fn+str(idx), item[:-4]),'wb'))
