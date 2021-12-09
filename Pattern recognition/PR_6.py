# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 10:27
# @Author  : Qianpeng Li
# @FileName: PR_6.py
# @Contact : liqianpeng2021@ia.ac.cn
import scipy.io as scio

from PR_6_tool import *

# 获得作业1的数据 X1 并生成相应的标签 Y1
path1 = 'X1.mat'
dic1 = scio.loadmat(path1)
X1 = dic1['X']
for i in range(5):
    if (i == 0):
        Y1 = np.zeros(200)
    else:
        Y1 = np.hstack((Y1, i * np.ones(200)))

# 获得作业2的数据 X2 并生成相应的标签 Y2
path2 = 'X2.mat'
dic2 = scio.loadmat(path2)
X2 = dic2['X']
for i in range(2):
    if (i == 0):
        Y2 = np.zeros(100)
    else:
        Y2 = np.hstack((Y2, i * np.ones(100)))

print('******************Assignment_1*********************\n')

kmeans = Kmeans(k=5, iter=100)
kmeans.fit(X1)
y_p1 = kmeans.labels
print('assignment1 kmeans ACC', ACC(Y1, y_p1),'\n')

# test part  用于比较本例程与sklearn-kmeans的精度

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=5, random_state=0).fit(X1)
# y_k = kmeans.labels_
# print('sklearn ACC', ACC(Y1, y_k))


print('******************Assignment_2*********************\n')
'''
ATTENTION!
method_w 只能为 'full_connection' or 'k_neighbor' or 'ep_thresh'
method_sc 只能为 'algorithm1' or 'algorithm2' or 'algorithm3'
'''
sc = Spectral_Clustering(k=2, k_neighbor=10, sigma=2, ep=0.75, method_w='k_neighbor', method_sc='algorithm3')
sc.fit(X2)
y_sc2 = sc.labels
print('assignment2 SpectralClustering ACC', ACC(Y2, y_sc2),'\n')

# test part  用于比较本例程与sklearn-SpectralClustering的精度

# from sklearn.cluster import SpectralClustering
# clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(X2)
# y_s2 = clustering.labels_
# print('SC ACC', ACC(Y2, y_s2))
