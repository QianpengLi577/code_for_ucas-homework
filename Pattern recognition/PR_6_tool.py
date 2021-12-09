# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 15:01
# @Author  : Qianpeng Li
# @FileName: PR_6_tool.py
# @Contact : liqianpeng2021@ia.ac.cn

import numpy as np
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


class Kmeans():
    '''
        Kmeans
    '''

    def __init__(self, k, iter):
        '''

        :param k: 类别数  一个实数
        :param iter: 最大迭代次数   一个实数

        '''
        self.k = k
        self.max_iter = iter
        self.center = None

    def generate_center(self, x):
        '''

        :param x: 聚类数据  shape：N*D  N为样本数  D为特征维度
        :return: u_list : 初始化的聚类中心，类型为list，len(list) = 类别数

        :process
            ① 随机挑选一个样本作为一个聚类中心，并添加到u_list
            ② 计算所有样本 到 现有的聚类中心 的欧氏距离  记为d_temp(N*m  N为样本数，m为现有聚类中心数)
            ③ 选取d_temp每一行的最小值构成新的d_temp 即获得每个样本到聚类中心的最小欧氏距离  min-step
               选择d_temp最大值所对应得样本 作为另一个聚类中心 并添加到u_list  max-step
            ④ 重复②③,直到获得k个聚类中心
        '''
        N = x.shape[0]  # 样本数
        u_list = []
        u1 = x[int(N * np.random.random_sample()), :].reshape((1, x.shape[1]))  # ① 随机获得一个聚类中心
        u_list.append(u1)
        for i in range(self.k - 1):  # 获得其余聚类中心
            d = []
            for u in u_list:
                d.append(np.linalg.norm(x - u, axis=1, keepdims=True))  # 计算所有样本到聚类中心得欧氏距离
            for j in range(len(d)):  # ②
                if (j == 0):
                    d_temp = d[0]
                else:
                    d_temp = np.hstack((d_temp, d[j]))

            d_temp = d_temp.min(1)  # 获得每个样本到聚类中心的最小欧氏距离  min-step
            index = np.argwhere(d_temp == d_temp.max())  # ③ 获得d_temp max对应得样本index  max-step
            # print(index[0,0])  # debug part 用于查看选择的初始聚类中心是否符合预期
            u_list.append(x[index[0, 0], :].reshape((1, x.shape[1])))
        return u_list

    def fit(self, x):
        '''

        :param x: 聚类数据  shape：N*D  N为样本数  D为特征维度
        :return: no return

        :process
            ① 通过 generate_center 获得初始的聚类中心
            ② 计算所有样本 到 聚类中心 的欧氏距离  记为d_temp(N*k  N为样本数，k为类别数)
            ③ 通过d_temp每行最小值的index给样本赋予标签
               通过标签对每一类求均值，重新获得聚类中心
            ④ 重复②③,直至运行 max_iter 次
        '''
        u_list = self.generate_center(x)  # 获得初始聚类中心
        y = np.zeros(x.shape[0])
        for i in range(self.max_iter):
            d = []
            for u in u_list:
                d.append(np.linalg.norm(x - u, axis=1, keepdims=True))
            for j in range(len(d)):
                if (j == 0):
                    d_temp = d[0]
                else:
                    d_temp = np.hstack((d_temp, d[j]))  # 获得所有样本到所有聚类中心的欧氏距离
            for j in range(x.shape[0]):
                y[j] = d_temp[j, :].argmin()  # 分配每个样本的类别
            u_list = []
            for j in range(self.k):
                idx = np.argwhere(y == j)
                x_temp = x[idx, :]
                u_list.append(x_temp.mean(0))  # 更新聚类中心
        self.center = u_list  # 将聚类中心赋值给 center
        self.get_label(x)  # 获得聚类结果

    def get_label(self, x):
        '''

        :param x: 聚类数据  shape：N*D  N为样本数  D为特征维度
        :return: no return

        :process
            ① 计算所有样本 到 聚类中心 的欧氏距离  记为d_temp(N*k  N为样本数，k为类别数)
            ② 通过d_temp每行最小值的index给样本赋予标签
        '''
        d = []
        y = np.zeros(x.shape[0])
        for u in self.center:
            d.append(np.linalg.norm(x - u, axis=1, keepdims=True))  # 计算样本到每个聚类中心的欧氏距离
        for j in range(len(d)):
            if (j == 0):
                d_temp = d[0]
            else:
                d_temp = np.hstack((d_temp, d[j]))
        for j in range(x.shape[0]):
            y[j] = d_temp[j, :].argmin()  # 对样本分配标签
        self.labels = y  # 标签赋值给 labels


class Spectral_Clustering():
    '''
    Spectral_Clustering
    '''

    def __init__(self, k, k_neighbor, sigma, ep, method_w, method_sc):
        '''

        :param k: 类别数 实数
        :param k_neighbor: k近邻构图的k  实数
        :param sigma: sigma of e^{ -\frac{ || x_{1}-x_{2} ||_{2}^{2} } { 2 \sigma^{ 2 } } }  实数
        :param ep: \epsilon of \epsilon neighborhood graph  实数
        :param method_w: 构图方式   仅为下方三种字符串
                        'full_connection'  -- 全连图
                        'k_neighbor'       -- K近邻图
                        'ep_thresh'        -- \epsilon neighborhood graph
        :param method_sc: 聚类的方法  仅为下方三种字符串
                        'algorithm1'       -- 经典算法
                        'algorithm2'       -- Shi 算法
                        'algorithm3'       -- Ng  算法
        '''
        self.k = k
        self.k_neighbor = k_neighbor
        self.sigma = sigma
        self.ep = ep
        self.method_w = method_w
        self.method_sc = method_sc

    def get_w(self, x):
        '''

        :param x: 聚类数据  shape：N*D  N为样本数  D为特征维度
        :return: 亲和矩阵  shape：N*N

        :process
            ① 计算点对亲和性
            ② 根据 method_w 获得相应的亲和矩阵
        '''
        d_list = []
        d = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            norm = np.linalg.norm(x - x[i, :].reshape((1, x.shape[1])), axis=1, keepdims=True)  # 计算样本点之间的欧氏距离
            d_list.append(norm)
            d[i, :] = norm.reshape((1, x.shape[0]))

        w_temp = np.exp(-d ** 2 / 2 / self.sigma ** 2)  # 获得点对的亲和矩阵

        if (self.method_w == 'full_connection'):  # 全连图
            return w_temp * (np.ones((x.shape[0], x.shape[0])) - np.eye(x.shape[0]))  # 将样本与自己的亲和性置为0

        elif (self.method_w == 'k_neighbor'):  # Knn 图
            mask = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                mask[i, (-w_temp[i, :]).argsort()[1:self.k_neighbor + 1]] = 1  # 获得每个样本的k_neighbor index
            return mask * w_temp

        elif (self.method_w == 'ep_thresh'):  # \epsilon neighborhood graph
            mask = (w_temp > self.ep).astype(np.int64)  # 获得亲和性大于 \epsilon 的index
            return mask * w_temp * (np.ones((x.shape[0], x.shape[0])) - np.eye(x.shape[0]))  # 将样本与自己的亲和性置为0
        else:
            print('key words error')
            return None

    def fit(self, x):
        '''

        :param x: 聚类数据  shape：N*D  N为样本数  D为特征维度
        :return: no return

        :process
            ① 通过get_w 获得亲和矩阵
            ② 根据 method_sc 对数据进行聚类
        '''
        w = self.get_w(x)  # 获得亲和矩阵
        W = (w + w.T) / 2  # 保证亲和矩阵对称
        diag = W.sum(axis=1)  #
        D = np.diag(diag)  # 获得度矩阵
        L = D - W  # 获得拉普拉斯矩阵
        clf = Kmeans(self.k, 100)  # 初始化kmeans

        if (self.method_sc == 'algorithm1'):  # 经典算法
            value, vector = np.linalg.eig(L)
            value = np.real(value)
            vector = np.real(vector)
            U = vector[:, value.argsort()[0:self.k]]
            clf.fit(U)
            self.labels = clf.labels

        elif (self.method_sc == 'algorithm2'):  # Shi 算法
            value, vector = np.linalg.eig(np.linalg.inv(D).dot(L))
            value = np.real(value)
            vector = np.real(vector)
            U = vector[:, value.argsort()[0:self.k]]
            clf.fit(U)
            self.labels = clf.labels

        elif (self.method_sc == 'algorithm3'):  # Ng 算法
            value, vector = np.linalg.eig(np.dot(np.sqrt(np.linalg.inv(D)), L.dot(np.sqrt(np.linalg.inv(D)))))
            value = np.real(value)
            vector = np.real(vector)
            U = vector[:, value.argsort()[0:self.k]]
            norm = np.linalg.norm(U, axis=1, keepdims=True)
            U = U / norm  # 归一化过程
            clf.fit(U)
            self.labels = clf.labels
        else:
            print('keywords error')
