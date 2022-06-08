from keras.datasets import mnist
import random
from tools import *
import warnings

warnings.filterwarnings("ignore")

# os win11 22458.1000

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 加载mnist原始数据集
# class_list = random.sample(range(y_train.min(), y_train.max() + 1), 2)  #获得随机两类的标签

class_list = [0, 1]
#################################################################
# 分离两类测试数据、标签，并将数据reshape为（1，28*28），除以255归一化到[0,1]
y_test_2 = np.uint8([])
x_test_2 = np.ones((1, 28, 28), dtype=np.uint8)
for loop in range(y_test.shape[0]):
    if y_test[loop] == class_list[0] or y_test[loop] == class_list[1]:
        y_test_2 = np.append(y_test_2, y_test[loop].astype(np.uint8))
        x_test_2 = np.append(x_test_2, np.expand_dims(x_test[loop], 0), axis=0)
x_test_2 = np.delete(x_test_2, 0, 0)
x_test_2 = x_test_2.reshape((x_test_2.shape[0], x_test_2.shape[1] * x_test_2.shape[2])).astype(np.float32)
x_test_2 = x_test_2 / 255.0
y_test_2 = (y_test_2 == class_list[1])  # label为class_list[1]重新定义为1，label为class_list[0]定义为0
y_test_2 = y_test_2.astype(np.float32)
##################################################################
# 分离两类训练数据、标签，并将数据reshape为（1，28*28），除以255归一化到[0,1]
y_train_2 = np.uint8([])
x_train_2 = np.ones((1, 28, 28), dtype=np.uint8)
for loop in range(y_train.shape[0]):
    if y_train[loop] == class_list[0] or y_train[loop] == class_list[1]:
        y_train_2 = np.append(y_train_2, y_train[loop].astype(np.uint8))
        x_train_2 = np.append(x_train_2, np.expand_dims(x_train[loop], 0), axis=0)
x_train_2 = np.delete(x_train_2, 0, 0)
x_train_2 = x_train_2.reshape((x_train_2.shape[0], x_train_2.shape[1] * x_train_2.shape[2])).astype(np.float32)
x_train_2 = x_train_2 / 255.0
y_train_2 = (y_train_2 == class_list[1])  # label为class_list[1]重新定义为1，label为class_list[0]定义为0
y_train_2 = y_train_2.astype(np.float32)

len_tr = len(y_train_2)
len_te = len(y_test_2)  # 获得训练数据、预测数据的样本数

x_train_2 = np.hstack((np.ones((len_tr, 1)), x_train_2))
x_test_2 = np.hstack((np.ones((len_te, 1)), x_test_2))
y_train_2 = y_train_2.reshape((y_train_2.shape[0], 1))
y_test_2 = y_test_2.reshape((y_test_2.shape[0], 1))  # 将训练数据和预测数据增加一维特征‘1’，用于LDA的w0项

####################################################
####################################################
#      LDF--逻辑斯蒂回归
lr = 0.01  # 学习率
iter = 300  # 迭代次数
w = np.zeros((x_train_2.shape[1], 1))  # w初试化

train(x_train_2, y_train_2, w, lr, iter)  # 训练
y_p = predict(x_test_2, w)  # 测试

print('logistic_acc:', ACC(y_p.astype(np.int64), y_test_2.astype(np.int64)))

####################################################
####################################################
#      QDF

x0 = x_train_2[np.argwhere(y_train_2 == 0)[:, 0]]  # 获得label=0的训练数据
x1 = x_train_2[np.argwhere(y_train_2 == 1)[:, 0]]  # 获得label=1的训练数据
# x0=x0.T
# x1=x1.T

xte0 = x_test_2[np.argwhere(y_test_2 == 0)[:, 0]]  # 获得label=0的测试数据
xte1 = x_test_2[np.argwhere(y_test_2 == 1)[:, 0]]  # 获得label=1的测试数据

p1 = (np.sum(y_train_2)) / y_train_2.shape[0]  # label==1训练数据的概率p1
p0 = 1 - p1  # label==0训练数据的概率p0
m0 = x0.mean(0)  # x0均值
m1 = x1.mean(0)  # x1均值
x0q = x0 - m0
x1q = x1 - m1
# cov0 = np.cov(x0q.T)
# cov1 = np.cov(x1q.T)
cov0 = np.dot(x0q.T, x0q) / x0q.shape[0]
cov1 = np.dot(x1q.T, x1q) / x1q.shape[0]  # 求协方差阵

e0, v0 = np.linalg.eig(cov0)  # 求特征值和特征矩阵
e0 = e0.real
v0 = v0.real
e0[(abs(e0) < 0.00001) & (e0 >= 0)] = 0.00001
e0[(abs(e0) < 0.00001) & (e0 < 0)] = -0.00001  # 对特征值进行处理，方便求逆
n0 = v0.dot(np.diag(1 / e0).dot(v0.T))  # 求逆矩阵

e1, v1 = np.linalg.eig(cov1)  # 求特征值和特征矩阵
e1 = e1.real
v1 = v1.real
e1[(abs(e1) < 0.00001) & (e1 >= 0)] = 0.00001  # 对特征值进行处理，方便求逆
e1[(abs(e1) < 0.00001) & (e1 < 0)] = -0.00001
n1 = v1.dot(np.diag(1 / e1).dot(v1.T))  # 求逆矩阵

# n0 = np.linalg.pinv(cov0)                    #广义逆矩阵
# n1 = np.linalg.pinv(cov1)

q00 = np.diagonal(-0.5 * np.dot((xte0 - m0), n0.dot((xte0 - m0).T))) + np.log(p0)  # 0类样本在0类的似然度函数
q01 = np.diagonal(-0.5 * np.dot((xte0 - m1), n1.dot((xte0 - m1).T))) + np.log(p1)  # 0类样本在1类的似然度函数

q10 = np.diagonal(-0.5 * np.dot((xte1 - m0), n0.dot((xte1 - m0).T))) + np.log(p0)  # 1类样本在0类的似然度函数
q11 = np.diagonal(-0.5 * np.dot((xte1 - m1), n1.dot((xte1 - m1).T))) + np.log(p1)  # 1类样本在1类的似然度函数

yp0 = (q00 >= q01).astype(np.int64)  # 0类预测标签
yp1 = (q10 <= q11).astype(np.int64)  # 1类预测标签
print('QDF_ACC', (sum(yp0) + sum(yp1)) / y_test_2.shape[0])  # 统计acc

##########################################################
##########################################################
#           sklearn-QDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_2, y_train_2)
y_pp = qda.predict(x_test_2)
print('SKlearn_acc', ACC(y_pp.astype(np.int64), y_test_2.astype(np.int64)))
