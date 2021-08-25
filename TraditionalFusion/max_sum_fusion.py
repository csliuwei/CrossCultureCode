import numpy as np 
import os
import pickle 


def val_cal(num1, num2):
    # num1: eeg values
    # num2: ecg values
    tmp1 = np.abs(num1)
    tmp2 = np.abs(num2)
    sss = tmp1 + tmp2
    proba1 = tmp1 / sss
    proba2 = tmp2 / sss
    if num1 < num2:
        return ((proba1, 0), (0, proba2))
    else:
        return ((0, proba1), (proba2, 0))

def making_proba(list1, list2):
    # list1: eeg values
    # list2: ecg values
    try:
        length = len(list1)
    except:
        length = list1.shape[0]
    eeg_proba = np.zeros((length, 2))
    ecg_proba = np.zeros((length, 2))
    for i in range(length):
        if list1[i]>=0:
            if list2[i] >=0:
                eeg_proba[i,0] = 0
                eeg_proba[i,1] = 1
                ecg_proba[i,0] = 0
                ecg_proba[i,1] = 1
            else:
                res = val_cal(list1[i], list2[i])
                eeg_proba[i,0] = res[0][0]
                eeg_proba[i,1] = res[0][1]
                ecg_proba[i,0] = res[1][0]
                ecg_proba[i,1] = res[1][1]
        else:
            if list2[i] >= 0:
                res = val_cal(list1[i], list2[i])
                eeg_proba[i,0] = res[0][0]
                eeg_proba[i,1] = res[0][1]
                ecg_proba[i,0] = res[1][0]
                ecg_proba[i,1] = res[1][1]
            else:
                eeg_proba[i,0] = 1
                eeg_proba[i,1] = 0
                ecg_proba[i,0] = 1
                ecg_proba[i,1] = 0
    return eeg_proba, ecg_proba

def max_fusion(array1, array2):
    row, col = array1.shape 
    p_label = []
    for i in range(row):
        tmp = np.vstack((array1[i,:], array2[i,:]))
        tmp = np.max(tmp, axis=0)
        p_label.append(np.argmax(tmp))
    return p_label 

def sum_fusion(array1, array2):
    row, col = array1.shape 
    p_label = []
    for i in range(row):
        tmp = array1[i,:] + array2[i,:]
        p_label.append(np.argmax(tmp))
    return p_label 

def cal_acc(list1, list2):
    length = len(list1)
    correct = 0
    for i in range(length):
        if int(list1[i]) == int(list2[i]):
            correct += 1
    return correct/length 



eeg_decision_val_dir = './decision_dir/de_LDS/'
eye_decision_val_dir = './decision_dir/eye_baseline/'
res_dir = './02_res_max/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

file_names = os.listdir(eeg_decision_val_dir)
file_names.sort()

res_cv = {}
for item in file_names:
    print(item)
    eeg_used = pickle.load(open(eeg_decision_val_dir+item, 'rb'))
    eye_used = pickle.load(open(eye_decision_val_dir+item, 'rb'))

    res_cv['fused_label'] = []
    res_cv['fused_acc'] = []
    eeg_decision_val = eeg_used['decision_val']
    eye_decision_val = eye_used['decision_val']
    test_label = eeg_used['test_label']

    p_label = max_fusion(eeg_decision_val, eye_decision_val)
    p_acc = cal_acc(p_label, test_label)
    res_cv['fused_label'].append(p_label)
    res_cv['fused_acc'].append(p_acc)

    pickle.dump(res_cv, open(res_dir+item, 'wb'))

