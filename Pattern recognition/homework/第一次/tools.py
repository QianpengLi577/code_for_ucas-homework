import numpy as np
from scipy.optimize import linear_sum_assignment
# scipy                              1.7.1
# numpy                              1.20.2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # sigmoid函数


def cost_function(x, y, w):
    num = len(y)
    cost = (-1.0 / num) * (y.T.dot(np.log(sigmoid(x.dot(w)))) + (1 - y).T.dot(np.log(1 - sigmoid(x.dot(w)))))  # 代价函数
    return cost


def train(x, y, w, lr, iter):
    num = len(y)
    for i in range(iter):
        w -= (lr / num) * x.T.dot((sigmoid(x.dot(w)) - y))  # 梯度下降


def predict(x, w):
    return np.round(sigmoid(np.dot(x, w)))  # 四舍五入用于两类分类


def ACC(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size  # 使用匈牙利算法求得真实的准确度
