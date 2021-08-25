import matlab.engine
import numpy as np 
import os
import pickle 

def softmax(array):
    row, col = array.shape
    new_array = np.zeros((row, col))
    for i in range(row):
        tmp = np.exp(array[i,:]) / np.sum( np.exp(array[i,:]))
        new_array[i,:] = tmp
    return new_array

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

def one_hot(label):
    row = len(label)
    new_label = np.zeros((row, 3))
    for i in range(row):
        new_label[i, int(label[i])] = 1
    return new_label

def fuzzy_integral(sample_num, all_prob, Label, eeg_prob, ecg_prob, class_num):
    fuzzy_prob = np.zeros((sample_num, class_num))
    fuzzy_mu = eng.MCalFuzzyMeasure(all_prob, Label, nargout=3)
    fuzzy_mu = np.asarray(fuzzy_mu[0])
    for k in range(class_num):
        for jj in range(sample_num):
            if eeg_prob[jj,k] < ecg_prob[jj,k]:
                fuzzy_prob[jj,k] = eeg_prob[jj,k] + fuzzy_mu[2*k+1] * (ecg_prob[jj, k] - eeg_prob[jj, k])
            else:
                fuzzy_prob[jj,k] = ecg_prob[jj,k] + fuzzy_mu[k] * (eeg_prob[jj,k] - ecg_prob[jj,k])
    return fuzzy_prob

# 
eng = matlab.engine.start_matlab()

eeg_decision_val_dir = './decision_dir/de_LDS/'
eye_decision_val_dir = './decision_dir/eye_baseline/'
res_dir = './03_res_fuzzy/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


file_names = os.listdir(eeg_decision_val_dir)
file_names.sort()

for item in file_names:
    print(item)
    eeg_used = pickle.load(open(eeg_decision_val_dir+item, 'rb'))
    eye_used = pickle.load(open(eye_decision_val_dir+item, 'rb'))

    res_cv = {}
    res_cv['fuzzy_proba'] = []
    res_cv['fused_label'] = []
    res_cv['fused_acc'] = []
    eeg_proba = eeg_used['decision_val']
    eye_proba = eye_used['decision_val']
    eeg_proba = softmax(eeg_proba)
    eye_proba = softmax(eye_proba)
    test_label = eeg_used['test_label']
    one_hot_label = one_hot(test_label)
    print(one_hot_label.shape)

    sample_num, prob_col = eeg_proba.shape
    all_proba = np.zeros((2, sample_num, prob_col))
    all_proba[0, :, :] = eeg_proba
    all_proba[1, :, :] = eye_proba
    all_proba = matlab.double(all_proba.tolist())
    Label = matlab.double(one_hot_label.tolist())

    p_label = fuzzy_integral(sample_num, all_proba, Label, eeg_proba, eye_proba, prob_col)
    fuzzy_p_label = []
    for hh in range(sample_num):
        tmp = np.argmax(p_label[hh, :])
        fuzzy_p_label.append(tmp)
    p_acc = cal_acc(fuzzy_p_label, test_label)
    res_cv['fuzzy_proba'] = p_label
    res_cv['fused_label'] = p_label
    res_cv['fused_acc'] = p_acc

    pickle.dump(res_cv, open(res_dir+item, 'wb'))

