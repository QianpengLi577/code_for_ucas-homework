# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/4 10:33
@Author  : Qianpeng Li
@FileName: ML_1.py
@Contact : liqianpeng2021@ia.ac.cn
"""
import matplotlib.pyplot as plt
import numpy as np

from Tools import *

x_train, y_train = get_train_data([2, 2], np.diag([0.5, 0.5]), [-2, -2], np.diag([1, 2]), 100)
x_test, y_test = get_test_data([2, 2], np.diag([0.5, 0.5]), [-2, -2], np.diag([1, 2]), 10)

# LDA part
clf_lda = LDA()
clf_lda.train_transform(x_train.T, y_train)
yp = clf_lda.predict(x_test.T)
acc = ACC(y_test, yp.T)
print('ACC_LDA:', acc)
W_LDA,B_LDA=clf_lda.get_parm()

# LR part
y_train_lr = y_train.copy()
y_train_lr[0:100] = -1
W_LR = LR(x_train.T, y_train_lr)
yp_lr = LR_predict(x_test.T, W_LR)
acc_lr = ACC(y_test, yp_lr.T)
print('ACC_LR:', acc_lr)
print('finsh')

# plot distribution
plt.figure(dpi=300,figsize=(6,6))
plt.title('data distribution')
plt.plot(x_train[0:100, 0], x_train[0:100, 1], 'ro', label='class1')
plt.plot(x_train[100:200, 0], x_train[100:200, 1], 'bs', label='class2')
plt.plot(x_test[:, 0], x_test[:, 1], 'g^', label='test')
plt.legend()
plt.show()

# 获得投影面、决策面
x_plot=np.linspace(min(x_train[:,0].min(),x_test[:,0].min()), max(x_train[:,0].max(),x_test[:,0].max()), num=100)
y_lda=(B_LDA-W_LDA[0]*x_plot)/W_LDA[1]
y_lr=-(W_LR[2]+W_LR[0]*x_plot)/W_LR[1]
y_proj=(W_LR[1]*x_plot)/W_LR[0]

x_range = (min(x_train[:,0].min(),x_test[:,0].min()),max(x_train[:,0].max(),x_test[:,0].max()))
y_range = (min(x_train[:,1].min(),x_test[:,1].min()),max(x_train[:,1].max(),x_test[:,1].max()))
range_f = max(max(abs(np.array(x_range))),max(abs(np.array(y_range))))

# lda 决策面
plt.figure(dpi=300,figsize=(6,6))
plt.title('lda_classification')
plt.plot(x_train[0:100, 0], x_train[0:100, 1], 'ro', label='class1')
plt.plot(x_train[100:200, 0], x_train[100:200, 1], 'bs', label='class2')
plt.plot(x_test[:, 0], x_test[:, 1], 'g^', label='test')
plt.plot(x_plot,y_lda,'r--')
plt.xlim((-range_f,range_f))
plt.ylim((-range_f,range_f))
plt.show()

# LR 决策面
plt.figure(dpi=300,figsize=(6,6))
plt.title('lr_classification')
plt.plot(x_train[0:100, 0], x_train[0:100, 1], 'ro', label='class1')
plt.plot(x_train[100:200, 0], x_train[100:200, 1], 'bs', label='class2')
plt.plot(x_test[:, 0], x_test[:, 1], 'g^', label='test')
plt.plot(x_plot,y_lr,'r--')
plt.xlim((-range_f,range_f))
plt.ylim((-range_f,range_f))
plt.show()

# LR 投影面
plt.figure(dpi=300,figsize=(6,6))
plt.title('Projection plane')
plt.plot(x_train[0:100, 0], x_train[0:100, 1], 'ro', label='class1')
plt.plot(x_train[100:200, 0], x_train[100:200, 1], 'bs', label='class2')
plt.plot(x_test[:, 0], x_test[:, 1], 'g^', label='test')
plt.plot(x_plot,y_proj,'r--')
plt.xlim((-range_f,range_f))
plt.ylim((-range_f,range_f))
plt.show()



