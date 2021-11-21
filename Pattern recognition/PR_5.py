# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 16:04
# @Author  : Qianpeng Li
# @FileName: PR_5.py
# @Contact : liqianpeng2021@ia.ac.cn

######################
######################
### LDA class
class LDA():
    def __init__(self, k):
        # k 实数
        self.w = None
        self.k = k

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
        # y_train n*1 样本的label

        # 输出参数
        # x_ k*n 降维后的数据

        y_temp = y_train.copy()
        y_temp = y_temp - y_train.min()
        length = y_train.max() - y_train.min() + 1
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
            # U[:,i]=m_
            Sb += N[i] * np.dot(m_ - m, (m_ - m).T)
            C = np.linalg.inv((Sw + 0.00001 * np.eye(Sw.shape[0]))).dot(Sb) / X.shape[
                1]  # Sw^{-1}*Sb 为了防止Sw^{-1}奇异，修改为(Sw+0.00001I)^{-1}*Sb
        value, vector = np.linalg.eig(C)
        value = np.real(value)
        vector = np.real(vector)  # 计算过程可能会出现复数，由于实对称阵特征值、特征向量为实数，因此这里直接取实部
        index = np.argsort(-value)
        self.w = vector[:, index[0:self.k]]
        x_ = np.dot(self.w.T, X)
        return x_

    def predict(self, X):
        # 功能
        # 降维数据

        # X d*n d为特征数  n为样本数  d需要与train的d一致

        x = X
        return np.dot(self.w.T, x)


######################
######################
### PCA class
class PCA():
    def __init__(self, k):
        # k 实数  目的维度
        self.k = k
        self.w = None
        self.m = None

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

    def train_transform(self, X):
        # 功能
        # 训练并降维训练数据

        # 输入参数
        # X d*n d样本维度 n样本个数

        # 输出参数
        # x_ k*n 降维后的数据

        x, self.m = self.mean_proc(X)
        c = x.dot(x.T) / X.shape[1]
        value, vector = np.linalg.eig(c)
        value = np.real(value)
        vector = np.real(vector)  # 计算过程可能会出现复数，由于实对称阵特征值、特征向量为实数，因此这里直接取实部
        index = np.argsort(-value)
        self.w = vector[:, index[0:self.k]]
        x_ = np.dot(self.w.T, x)
        return x_

    def predict(self, X):
        # 功能
        # 降维数据

        # X d*n d为特征数  n为样本数  d需要与train的d一致

        x = X - self.m
        return np.dot(self.w.T, x)


########################
########################
#### KNN function
def KNN_predict(x_train, x_test, y_train, k):
    # 功能
    # KNN 分类

    # 输入参数
    # x_train d*n d为特征维度 n为样本数
    # x_test  d*m d为特征维度 m为样本数
    # y_train n*1 n为x_train样本数
    # k 实数，KNN的k

    # 输出参数
    # y_predict m*1 预测的标签

    N_test = x_test.shape[1]
    y_temp = []
    y_predict = np.zeros(N_test)
    for i in range(N_test):
        x_temp = x_train - x_test[:, i].reshape((x_test.shape[0], 1))
        norm = np.linalg.norm(x_temp, ord=2, axis=0, keepdims=False)  # 计算测试样本与训练的二范数
        index = np.argsort(norm)[0:k]  # 寻找k个最短norm的index
        c = y_train[index]  # index所对应的类别
        y_temp.append(np.argmax(norm))
        y_predict[i] = np.argmax(np.bincount(c))  # 寻找类别数最多的bin
    return y_predict


######################
######################
### ACC function
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


import numpy as np
import scipy.io as scio
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

#  load and split orl data
path_orl = 'ORLData_25.mat'

dic_orl = scio.loadmat(path_orl)
orl = dic_orl['ORLData']
dim, N = orl.shape
data_orl = orl[0:dim - 1, :].astype(float)
label_orl = orl[dim - 1, :]

x_train, x_test, y_train, y_test = train_test_split(data_orl.T, label_orl, test_size=0.2, random_state=1)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.reshape(-1)

k_nn = 1  # knn参数
print('orl part')
orlrecord = np.zeros((20, 2))
for i in range(20):
    k_orl = 5 * (i + 1)
    print('k_orl=', k_orl)

    # orl using LDA
    clf_lda = LDA(k_orl)
    x_train_lda = clf_lda.train_transform(x_train, y_train)
    x_test_lda = clf_lda.predict(x_test)
    y_predict = KNN_predict(x_train_lda, x_test_lda, y_train, k_nn)
    print('knn_' + str(k_nn) + '_lda:', ACC(y_test, y_predict))
    orlrecord[i, 0] = ACC(y_test, y_predict)

    # orl using PCA
    clf_pca = PCA(k_orl)
    x_train_pca = clf_pca.train_transform(x_train)
    x_test_pca = clf_pca.predict(x_test)
    y_predict = KNN_predict(x_train_pca, x_test_pca, y_train, k_nn)
    print('knn_' + str(k_nn) + '_pca:', ACC(y_test, y_predict))
    orlrecord[i, 1] = ACC(y_test, y_predict)

###################################################33
#####################################################
####################### vec part ###################

#   load and split veh data
path_veh_x = 'vehicle_x.mat'
path_veh_y = 'vehicle_y.mat'
dic_veh_x = scio.loadmat(path_veh_x)
dic_veh_y = scio.loadmat(path_veh_y)
data_veh = dic_veh_x['X'].astype(float)
label_veh = dic_veh_y['labels']

x_train, x_test, y_train, y_test = train_test_split(data_veh, label_veh, test_size=0.2, random_state=1)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.reshape(-1)
print('\n')
print('veh part')
vehrecord = np.zeros((10, 2))
for i in range(10):
    k_veh = 2 + i
    print('k_veh=', k_veh)

    # veh using LDA
    clf_lda = LDA(k_veh)
    x_train_lda = clf_lda.train_transform(x_train, y_train)
    x_test_lda = clf_lda.predict(x_test)
    y_predict = KNN_predict(x_train_lda, x_test_lda, y_train, k_nn)
    print('knn_' + str(k_nn) + '_lda:', ACC(y_test, y_predict))
    vehrecord[i, 0] = ACC(y_test, y_predict)

    # veh using PCA
    clf_pca = PCA(k_veh)
    x_train_pca = clf_pca.train_transform(x_train)
    x_test_pca = clf_pca.predict(x_test)
    y_predict = KNN_predict(x_train_pca, x_test_pca, y_train, k_nn)
    print('knn_' + str(k_nn) + '_pca:', ACC(y_test, y_predict))
    vehrecord[i, 1] = ACC(y_test, y_predict)

np.savetxt('orlrecord.csv', orlrecord, fmt='%.6f', delimiter=',')
np.savetxt('vehrecoed.csv', vehrecord, fmt='%.6f', delimiter=',')
print('finsh')
