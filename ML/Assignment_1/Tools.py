# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/4 9:35
@Author  : Qianpeng Li
@FileName: Tools.py
@Contact : liqianpeng2021@ia.ac.cn
"""
import numpy as np
import matplotlib.pyplot as plt


# 生成训练数据集，每类100个数据
def get_train_data(mu_1, cov_1, mu_2, cov_2, data_size=100):
    # class 1
    data_train1 = np.random.multivariate_normal(mu_1, cov_1, data_size)
    data_label = np.zeros((2 * data_size, 1))
    data_label[0:data_size, :] = 0
    # class 2
    data_train2 = np.random.multivariate_normal(mu_2, cov_2, data_size)
    data_train = np.concatenate((data_train1, data_train2), axis=0)
    data_label[data_size:2 * data_size, :] = 1

    return data_train, data_label


# 生成测试数据集，每类10个
def get_test_data(mu_1, cov_1, mu_2, cov_2, data_size=10):
    # class 1
    data_test1 = np.random.multivariate_normal(mu_1, cov_1, data_size)
    data_label = np.zeros((2 * data_size, 1))
    data_label[0:data_size, :] = 0
    # class 2
    data_test2 = np.random.multivariate_normal(mu_2, cov_2, data_size)
    data_test = np.concatenate((data_test1, data_test2), axis=0)
    data_label[data_size:2 * data_size, :] = 1

    return data_test, data_label


######################
######################
### LDA class
class LDA():
    def __init__(self):
        # w 投影矩阵
        # mu_list 存储类均值

        self.w = None
        self.mu_list = None

    def mean_proc(self, X):
        # 功能
        # 零均值化过程

        # 输入参数
        # X d*n d为维度  n为样本个数

        # 输出参数
        # X-m 零均值样本
        # m 样本均值

        m = X.mean(axis=1)
        m = m.reshape((len(m), 1))
        return X - m, m

    def train_transform(self, X, y_train):
        # 功能
        # 训练LDA并将训练数据降维

        # 输入参数
        # X d*n d样本维度 n样本个数
        # y_train n*1 样本的label  样本标签必须为连续的  比如0 1 2是可行的，-1 1 3是不可行的

        # 输出参数
        # x_ k*n 降维后的数据

        y_temp = y_train.copy()
        y_temp = y_temp - y_train.min()
        length = int(y_train.max() - y_train.min()) + 1
        N = np.zeros(length)
        U = np.zeros((X.shape[0], length))
        Sw = np.zeros((X.shape[0], X.shape[0]))
        Sb = np.zeros((X.shape[0], X.shape[0]))
        x__, m = self.mean_proc(X)
        for i in range(length):
            # 分别获得每类的均值、零均值样本、每类样本数，计算Sw Sb
            x_, m_ = self.mean_proc(X[:, np.argwhere(y_temp == i).reshape(-1)])
            N[i] = len(np.argwhere(y_temp == i))
            Sw += x_.dot(x_.T)
            U[:, i] = m_.reshape(-1)
            Sb += N[i] * np.dot(m_ - m, (m_ - m).T)
            C = np.linalg.inv((Sw + 0.00001 * np.eye(Sw.shape[0]))).dot(Sb) / X.shape[
                1]  # Sw^{-1}*Sb 为了防止Sw^{-1}奇异，修改为(Sw+0.00001I)^{-1}*Sb
        value, vector = np.linalg.eig(C)
        value = np.real(value)
        vector = np.real(vector)  # 计算过程可能会出现复数，由于实对称阵特征值、特征向量为实数，因此这里直接取实部
        index = np.argsort(-value)
        self.w = vector[:, index[0:length - 1]]
        self.mu_list = U
        x_ = np.dot(self.w.T, X)
        return x_

    def predict(self, X):
        # 功能
        # 预测   !!!!  当且仅当两类时才能使用

        # X d*n d为特征数  n为样本数  d需要与train的d一致

        x = X
        mu_mid = np.dot(self.w.T, self.mu_list)
        thresh = mu_mid.mean()
        z = np.dot(self.w.T, x)
        yp = (z > thresh).astype(int)
        return yp

    def get_parm(self):
        mu_mid = np.dot(self.w.T, self.mu_list)
        thresh = mu_mid.mean()
        return self.w,thresh


######################
######################
### LR function
def LR(X, Y):
    # 功能
    # 线性回归

    # X d*n d为特征数  n为样本数
    # Y n*1 n为样本数  样本标签必须为连续的  比如0 1 2是可行的，-1 1 3是不可行的
    N = len(Y)
    X_new = np.concatenate((X, np.ones((1, N))), axis=0)
    la = 0.000001
    W = np.linalg.inv(X_new.dot(X_new.T) + la * np.eye(X_new.shape[0])).dot(X_new).dot(Y)
    return W

### LR function
def LR_predict(X,W):
    # 功能
    # 线性回归

    # X d*n d为特征数  n为样本数
    # Y n*1 n为样本数  样本标签必须为连续的  比如0 1 2是可行的，-1 1 3是不可行的
    N = X.shape[1]
    X_new = np.concatenate((X, np.ones((1, N))), axis=0)
    yp = (np.dot(W.T,X_new)>0).astype(int)
    return yp


######################
######################
### ACC function
from scipy.optimize import linear_sum_assignment


def ACC(y_true, y_pred):
    # ACC func
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size  # 使用匈牙利算法求得真实的准确度
