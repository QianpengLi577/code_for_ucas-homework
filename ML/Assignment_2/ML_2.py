# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/10 15:35
@Author  : Qianpeng Li
@FileName: ML_2.py
@Contact : liqianpeng2021@ia.ac.cn
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Tools import *

def load_data():
    iris = datasets.load_iris()
    # 数据共 150 行 ，4 列
    # print(iris.data.shape)
    # 150 行数据 对应 150 个标签值（种类）
    # print(iris.target.shape)
    return iris.data, iris.target

# load data
x, y = load_data()
# one-hot
labels = np.zeros((150, 3))
for i in range(150):
    labels[i][y[i]] = 1
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, stratify=labels)
# 3-layer neural network  input 4 hidden 50 output 3  batch_size 30  epoch_max 10000 learning_rate 0.05 error for stop 0.01
modle = NN(4, 50, 3, 30, 10000, 0.05, 0.01)
modle.train(x_train, y_train, True)
print(modle.acc(x_test, y_test))

# plot
x_line = np.linspace(1, 10000, num=10000, endpoint=True)
plt.plot(x_line,modle.loss)
plt.title('loss-epoch')
plt.savefig('loss.png')
plt.show()
