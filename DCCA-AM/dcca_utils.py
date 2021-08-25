import os
import numpy as np
import pickle
import torch
import torch.nn as nn

def loading_cv_data(eeg_dir, eye_dir, file_name, cv_number):
    eeg_data_pickle = np.load( os.path.join(eeg_dir, file_name))
    eye_data_pickle = np.load( os.path.join(eye_dir, file_name))
    eeg_data = pickle.loads(eeg_data_pickle['data'])
    eye_data = pickle.loads(eye_data_pickle['data'])
    label = pickle.loads(eeg_data_pickle['label'])
    list_1 = [0,1,2,3,4,15,16,17,18,19,30,31,32,33,34]
    list_2 = [5,6,7,8,9,20,21,22,23,24,35,36,37,38,39]
    list_3 = [10,11,12,13,14,25,26,27,28,29,40,41,42,43,44]
    if cv_number == 1:
        print('#1 as test, preparing data')
        train_list = list_2 + list_3
        test_list = list_1
    elif cv_number == 2:
        print('#2 as test, preparing data')
        train_list = list_1 + list_3
        test_list = list_2
    else:
        print('#3 as test, preparing data')
        train_list = list_1 + list_2
        test_list = list_3

    train_eeg = []
    test_eeg = []
    train_label = []
    for train_id in range(len(train_list)):
        train_eeg_tmp = eeg_data[train_list[train_id]]
        train_eye_tmp = eye_data[train_list[train_id]]
        train_label_tmp = label[train_list[train_id]]
        if train_id == 0:
            train_eeg = train_eeg_tmp
            train_eye = train_eye_tmp
            train_label = train_label_tmp
        else:
            train_eeg = np.vstack((train_eeg, train_eeg_tmp))
            train_eye = np.vstack((train_eye, train_eye_tmp))
            train_label = np.hstack((train_label, train_label_tmp))
    assert train_eeg.shape[0] == train_eye.shape[0]
    assert train_eeg.shape[0] == train_label.shape[0]

    test_eeg = []
    test_eye = []
    test_label = []
    for test_id in range(len(test_list)):
        test_eeg_tmp = eeg_data[test_list[test_id]]
        test_eye_tmp = eye_data[test_list[test_id]]
        test_label_tmp = label[test_list[test_id]]
        if test_id == 0:
            test_eeg = test_eeg_tmp
            test_eye = test_eye_tmp
            test_label = test_label_tmp
        else:
            test_eeg = np.vstack((test_eeg, test_eeg_tmp))
            test_eye = np.vstack((test_eye, test_eye_tmp))
            test_label = np.hstack((test_label, test_label_tmp))
    assert test_eeg.shape[0] == test_eye.shape[0]
    assert test_eeg.shape[0] == test_label.shape[0]

    train_all = np.hstack((train_eeg, train_eye, train_label.reshape([-1,1])))
    test_all = np.hstack((test_eeg, test_eye, test_label.reshape([-1,1])))
    # print(train_all.shape)
    # print(test_all.shape)

    return train_all, test_all


def linear_cca_weight_calculation(H1, H2, outdim_size):
    weight = [None, None]
    mean_matrix = [None, None]

    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o1 = H1.shape[1]
    o2 = H2.shape[1]

    mean_matrix[0] = np.mean(H1, axis=0)
    mean_matrix[1] = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean_matrix[0], (m, 1))
    H2bar = H2 - np.tile(mean_matrix[1], (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = np.dot(
        np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(
        np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv,
                               SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    weight[0] = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    weight[1] = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]
    return weight, mean_matrix

class LinearCCATransform(nn.Module):
    def __init__(self, weight, matrix_mean, device):
        super(LinearCCATransform, self).__init__()
        weight_0 = torch.from_numpy(weight[0]).to(torch.float).to(device)
        weight_1 = torch.from_numpy(weight[1]).to(torch.float).to(device)
        self.w = [weight_0, weight_1] # tensor
        self.m = matrix_mean # numpy
        self.device = device

    def _get_resul(stelf, x, idx):
        result = x - torch.from_numpy(self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)).to(self.device)
        # breakpoint()
        # result = numpy.dot(result, self.w[idx])
        result = torch.matmul(result, self.w[idx])
        return result

    def forward(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)

def cca_metric_derivative(H1, H2):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9
    # transform the matrix: to be consistent with the original paper
    H1 = H1.T
    H2 = H2.T
    # o1 and o2 are feature dimensions
    # m is sample number
    o1 = o2 = H1.shape[0]
    m = H1.shape[1]

    # calculate parameters
    H1bar = H1 - H1.mean(axis=1).reshape([-1,1])
    H2bar = H2 - H2.mean(axis=1).reshape([-1,1])

    SigmaHat12 = (1.0 / (m - 1)) * np.matmul(H1bar, H2bar.T)
    SigmaHat11 = (1.0 / (m - 1)) * np.matmul(H1bar, H1bar.T) + r1 * np.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.matmul(H2bar, H2bar.T) + r2 * np.eye(o2)

    # eigenvalue and eigenvector decomposition
    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    # remove eighvalues and eigenvectors smaller than 0
    posInd1 = np.where(D1 > 0)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]

    posInd2 = np.where(D2 > 0)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    # calculate matrxi T
    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)
    Tval = np.matmul(np.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)
    # By default, we will use all the singular values
    tmp = np.matmul(Tval.T, Tval)
    #corr = np.trace(np.sqrt(tmp))
    corr = np.sqrt(np.trace(tmp))
    cca_loss = -1 * corr

    # calculate the derivative of H1 and H2
    U_t, D_t, V_prime_t = np.linalg.svd(Tval)
    Delta12 = SigmaHat11RootInv @ U_t @ V_prime_t @ SigmaHat22RootInv
    Delta11 = SigmaHat11RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat11RootInv
    Delta22 = SigmaHat22RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat22RootInv
    Delta11 = -0.5 * Delta11
    Delta22 = -0.5 * Delta22

    DerivativeH1 = ( 1.0 / (m - 1)) * (2 * (Delta11 @ H1bar) + Delta12 @ H2bar)
    DerivativeH2 = ( 1.0 / (m - 1)) * (2 * (Delta22 @ H2bar) + Delta12 @ H1bar)

    return cca_loss, DerivativeH1.T, DerivativeH2.T


class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        breakpoint()
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

class TransformLayers(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(TransformLayers, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    nn.Sigmoid(),
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id+1], affine=False),
                    ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, output_dim):
        super(AttentionFusion, self).__init__()
        self.output_dim = output_dim
        self.attention_weights = nn.Parameter(torch.randn(self.output_dim, requires_grad=True))
    def forward(self, x1, x2):
        # calculate weigths for all input samples
        row, _ = x1.shape
        fused_tensor = torch.empty_like(x1)
        alpha = []
        for i in range(row):
            tmp1 = torch.dot(x1[i,:], self.attention_weights)
            tmp2 = torch.dot(x2[i,:], self.attention_weights)
            alpha_1 = torch.exp(tmp1) / (torch.exp(tmp1) + torch.exp(tmp2))
            alpha_2 = 1 - alpha_1
            alpha.append((alpha_1.detach().cpu().numpy(), alpha_2.detach().cpu().numpy()))
            fused_tensor[i, :] = alpha_1 * x1[i,:] + alpha_2 * x2[i, :]
        return fused_tensor, alpha

class DCCA_AM(nn.Module):
    def __init__(self, input_size1, input_size2, layer_sizes1, layer_sizes2, outdim_size, categories, device):
        super(DCCA_AM, self).__init__()
        self.outdim_size = outdim_size
        self.categories = categories
        # self.use_all_singular_values = use_all_singular_values
        self.device = device

        self.model1 = TransformLayers(input_size1, layer_sizes1).to(self.device)
        self.model2 = TransformLayers(input_size2, layer_sizes2).to(self.device)

        self.model1_parameters = self.model1.parameters()
        self.model2_parameters = self.model1.parameters()

        self.classification = nn.Linear(self.outdim_size, self.categories)

        self.attention_fusion = AttentionFusion(outdim_size)
        # self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss # this is a function to calculate the negative of cca loss
    def forward(self, x1, x2):
        # forward process: returns negative of cca loss and predicted labels
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        # cca_loss_val = self.loss(output1, output2)
        cca_loss, partial_h1, partial_h2 = cca_metric_derivative(output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
        # breakpoint()
        #linear cca weights calculation
        # cca_input_1 = output1.detach().cpu().numpy()
        # cca_input_2 = output2.detach().cpu().numpy()
        # cca_weights, cca_mean_matrix = linear_cca_weight_calculation(cca_input_1, cca_input_2, self.outdim_size)
        # self.cca_extraction = LinearCCATransform(cca_weights, cca_mean_matrix, self.device)
        # transformed_1, transformed_2 = self.cca_extraction(output1, output2)
        # attention-based fusion
        # fused_tensor, alpha = self.attention_fusion(transformed_1, transformed_2)
        fused_tensor, alpha = self.attention_fusion(output1, output2)
        out = self.classification(fused_tensor)
        # return out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor.detach().cpu().data, transformed_1.detach().cpu().data, transformed_2.detach().cpu().data, alpha
        return out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor.detach().cpu().data, alpha
