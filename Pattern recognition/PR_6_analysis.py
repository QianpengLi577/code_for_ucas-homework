# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 15:27
# @Author  : Qianpeng Li
# @FileName: PR_6_analysis.py
# @Contact : liqianpeng2021@ia.ac.cn
import numpy as np
import scipy.io as scio
import warnings
warnings.filterwarnings("ignore")

from PR_6_tool import *


def permutation(nums, p, q, s):
    # 功能
    # 递归生成n!个排列数，用于生成矩阵列坐标index

    # 输入参数
    # nums 列坐标范围 list
    # p q 用于排列的数的范围  为两个实数
    # s list

    # 输出参数
    # 无 n!排列数直接存放在s内

    # eg
    # p=[]
    # nums = [i for i in range(3)]
    # permutation(nums, 0, len(nums), p)
    # print(p)
    # [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]

    if p == q:
        s.append(list(nums))
    else:
        for i in range(p, q):
            nums[i], nums[p] = nums[p], nums[i]
            permutation(nums, p + 1, q, s)
            nums[i], nums[p] = nums[p], nums[i]


def label2str(label):
    # 将label 转换为不同的颜色
    index = ['xkcd:purple', 'xkcd:blue', 'xkcd:brown', 'xkcd:red', 'xkcd:teal', 'xkcd:orange', 'xkcd:yellow',
             'xkcd:sky blue', 'xkcd:dark green', 'xkcd:dark blue', 'xkcd:cyan']
    sum = len(label)
    out = []
    for i in range(sum):
        out.append(index[label[i]])
    return out


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
frequency = 50
err_his = np.zeros(frequency)
acc_his = np.zeros(frequency)

p = []
nums = [i for i in range(5)]
permutation(nums, 0, len(nums), p)  # 获得预测得到的均值与真实的均值可能存在的映射关系
mu1 = np.array([1, -1])
mu2 = np.array([5.5, -4.5])
mu3 = np.array([1, 4])
mu4 = np.array([6, 4.5])
mu5 = np.array([9, 0.0])

mu_list = []  # mu_list
mu_list.append(mu1)
mu_list.append(mu2)
mu_list.append(mu3)
mu_list.append(mu4)
mu_list.append(mu5)

kmeans = Kmeans(k=5, iter=100)
for i in range(frequency):
    kmeans.fit(X1)
    y_p1 = kmeans.labels
    center = kmeans.center
    acc_his[i] = ACC(Y1, y_p1)  # 记录acc
    temp = np.zeros(len(p))
    for j in range(len(p)):
        sum = 0
        sum += np.linalg.norm(center[0].reshape(1, X1.shape[1]) - mu_list[p[j][0]].reshape(1, X1.shape[1]), axis=1,
                              keepdims=True)[0]
        sum += np.linalg.norm(center[1].reshape(1, X1.shape[1]) - mu_list[p[j][1]].reshape(1, X1.shape[1]), axis=1,
                              keepdims=True)[0]
        sum += np.linalg.norm(center[2].reshape(1, X1.shape[1]) - mu_list[p[j][2]].reshape(1, X1.shape[1]), axis=1,
                              keepdims=True)[0]
        sum += np.linalg.norm(center[3].reshape(1, X1.shape[1]) - mu_list[p[j][3]].reshape(1, X1.shape[1]), axis=1,
                              keepdims=True)[0]
        sum += np.linalg.norm(center[4].reshape(1, X1.shape[1]) - mu_list[p[j][4]].reshape(1, X1.shape[1]), axis=1,
                              keepdims=True)[0]
        temp[j] = sum
    err_his[i] = temp.min()  # 获得mu的误差

kmeans.fit(X1)
y_p1 = kmeans.labels
center = kmeans.center

import matplotlib.pyplot as plt

x_plot = np.linspace(1, frequency, frequency)
plt.plot(x_plot, acc_his, color='blue', label='$ACC$', linewidth=0.8)
plt.xlabel('time')
plt.ylabel('acc')
plt.title('Kmeans acc ')
plt.savefig('./result/assignment1_acc_update.png')
plt.show()
# acc picture

plt.plot(x_plot, err_his, color='red', label='$ERROR$', linewidth=0.8)
plt.xlabel('time')
plt.ylabel('error')
plt.title('Kmeans error of mu ')
plt.savefig('./result/assignment1_error_update.png')
plt.show()
# error picture

print('acc_max', acc_his.max())
print('acc_min', acc_his.min())
print('acc_mean', acc_his.mean())
print('error_max', err_his.max())
print('error_min', err_his.min())
print('error_mean', err_his.mean())
print('center',center)

from sklearn.manifold import TSNE

# tsne visualization

tsne = TSNE()
tsne.fit_transform(X1)
color = label2str(y_p1.astype(np.int64))
plt.xticks([])
plt.yticks([])
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=color, marker='o', s=15)
plt.title('Kmeans visualization')
plt.savefig('./result/Assignment1_visualization.png')
plt.show()

print('******************Assignment_2*********************\n')
'''
ATTENTION!
method_w 只能为 'full_connection' or 'k_neighbor' or 'ep_thresh'
method_sc 只能为 'algorithm1' or 'algorithm2' or 'algorithm3'
'''
knn_list=np.linspace(3, 20, 18)  # 3~20
sigma_list = np.linspace(0.1, 5.0, 50)# 0.1~5.0
sc_his = np.zeros((len(knn_list),len(sigma_list)))
for i in range(len(knn_list)):
    for j in range(len(sigma_list)):
        sc = Spectral_Clustering(k=2, k_neighbor=knn_list[i], sigma=sigma_list[j], ep=0.75, method_w='k_neighbor', method_sc='algorithm3')
        sc.fit(X2)
        # print('i:',i,'j',j,'finsh')
        y_sc2 = sc.labels
        sc_his[i,j]=ACC(Y2, y_sc2)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 9,
}
plt.plot(knn_list,sc_his[:,9],color='green', label='$sigma=1.0$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(knn_list,sc_his[:,19],color='blue', label='$sigma=2.0$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(knn_list,sc_his[:,29],color='black', label='$sigma=3.0$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(knn_list,sc_his[:,39],color='red', label='$sigma=4.0$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.xlabel('k-neighbor')
plt.ylabel('ACC')
plt.title('sc acc with different sigma and k')
plt.savefig('./result/sc_different_k.png')
plt.show()

plt.plot(sigma_list,sc_his[2,:],color='blue', label='$k=5$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(sigma_list,sc_his[7,:],color='red', label='$k=10$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(sigma_list,sc_his[12,:],color='green', label='$k=15$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.plot(sigma_list,sc_his[17,:],color='black', label='$k=20$', linewidth=0.8)
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.xlabel('sigma')
plt.ylabel('ACC')
plt.title('sc acc with different sigma and k')
plt.savefig('./result/sc_different_sigma.png')
plt.show()

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
x,y = np.meshgrid(knn_list, sigma_list)
surf = ax.plot_surface(x, y, sc_his.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('sc acc with different sigma and k')
plt.savefig('./result/3d.png')
plt.show()

# test part  用于比较本例程与sklearn-SpectralClustering的精度

# from sklearn.cluster import SpectralClustering
# clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(X2)
# y_s2 = clustering.labels_
# print('SC ACC', ACC(Y2, y_s2))
