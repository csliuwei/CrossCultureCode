import scipy.io as sio 
import numpy as np 
import os
from sklearn import preprocessing 
import pickle 

from utils import svm_classification, concat_process

def warn(*args, **kwargs):
    pass
import warnings 
warnings.warn = warn 


eeg_dir = '../Features/China/eeg_used_4s'
eeg_file_list = os.listdir(eeg_dir)

eye_dir = '../Features/China/eye_used/'

eeg_file_list.sort()

res_dir = './01_svm_concat/'
if not os.path.exists(res_dir):
    os.mkdir(res_dir)


for item in eeg_file_list:
    print(item)
    eeg_data = np.load( os.path.join(eeg_dir, item) )
    eye_data = sio.loadmat( os.path.join(eye_dir, item[:-4]+'.mat') )
    concat_train, concat_test, train_label, test_label = concat_process(eeg_data, eye_data)
    
    concat_train = preprocessing.scale(concat_train)
    concat_test = preprocessing.scale(concat_test)
    best_res = svm_classification(concat_train, concat_test, train_label, test_label)
    pickle.dump(best_res, open(os.path.join(res_dir, item[:-4]),'wb'))


