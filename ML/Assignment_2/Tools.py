# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/10 15:34
@Author  : Qianpeng Li
@FileName: Tools.py
@Contact : liqianpeng2021@ia.ac.cn
"""
import numpy as np
class NN():
    def __init__(self, input_shape, hidden_shape, output_shape, batch_size, epoch_max, lr, error):
        #         parameter
        # input_shape  输入数据的维度
        # hidden_shape 隐层结点数
        # output_shaoe  类别数
        # batch_size batch大小，需要能够被样本数整除
        # epoch_max 迭代最大次数
        # lr 学习率
        # error loss小于一定数值终止train过程

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epoch_max = epoch_max
        self.lr = lr
        self.error = error
        self.w1 = (np.random.rand(self.input_shape, self.hidden_shape) - 0.5) / 10  # 所有的权重和偏执初始化为-0.05~0.05
        self.b1 = (np.random.rand(1, self.hidden_shape) - 0.5) / 10
        self.w2 = (np.random.rand(self.hidden_shape, self.output_shape) - 0.5) / 10
        self.b2 = (np.random.rand(1, self.output_shape) - 0.5) / 10
        self.loss = np.zeros(epoch_max)
        self.loss_history = []

    def sigmoid(self, x):
        #         parameter
        # x  n*p numpy array

        # sigmoid函数
        return 1 / (1 + np.exp(-x))

    def Dsigmoid(self, x):
        #         parameter
        # x  n*p numpy array

        # sigmoid导函数
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def Dtanh(self, x):
        #         parameter
        # x  n*p numpy array

        # 双曲正切函数的导函数
        return 1 - np.square(np.tanh(x))

    def train(self, x, t, print_flag):
        #         parameter
        # x  n*d numpy array
        # t  经过one-hot编码的标签，n*c numpy array
        # print_flag  Ture or False，用于标记是否打印log

        # deep copy
        x1 = x.copy()
        for epoch in range(self.epoch_max):

            loss_epoch = 0
            for m in range(int(x1.shape[0] / self.batch_size)):
                # 前向过程
                net1 = np.dot(x1[m * self.batch_size:(m + 1) * self.batch_size, :], self.w1)  # 计算 net1

                y = np.tanh(net1 + self.b1)  # 激活

                net2 = np.dot(y, self.w2)  # 计算net2

                z = self.sigmoid(net2 + self.b2)  # 激活

                loss_epoch += 0.5 * np.sum(np.square(t[m * self.batch_size:(m + 1) * self.batch_size, :] - z))  # loss

                self.loss_history.append(
                    0.5 * np.sum(np.square(t[m * self.batch_size:(m + 1) * self.batch_size, :] - z)))
                # print('epoch:', epoch + 1, ' batch:', m + 1, ' loss:', loss)

                # 反向过程
                dw2 = self.lr * np.dot(y.T, self.Dsigmoid(net2 + self.b2) * (
                        t[m * self.batch_size:(m + 1) * self.batch_size, :] - z))
                db2 = self.lr * self.Dsigmoid(net2 + self.b2) * (
                        t[m * self.batch_size:(m + 1) * self.batch_size, :] - z)
                dw1 = self.lr * np.dot(x1[m * self.batch_size:(m + 1) * self.batch_size, :].T,
                                       self.Dtanh(net1 + self.b1) * (np.dot(self.Dsigmoid(net2 + self.b2) * (
                                               t[m * self.batch_size:(m + 1) * self.batch_size, :] - z),
                                                                            self.w2.T)))
                db1 = self.lr * self.Dtanh(net1 + self.b1) * (
                    np.dot(self.Dsigmoid(net2 + self.b2) * (t[m * self.batch_size:(m + 1) * self.batch_size, :] - z),
                           self.w2.T))  # 计算出dw1 dw2 db1 db2
                self.w2 = self.w2 + dw2
                self.b2 = self.b2 + db2
                self.w1 = self.w1 + dw1
                self.b1 = self.b1 + db1  # 梯度下降
            if print_flag: print('epoch:', epoch, ' loss_avg:', loss_epoch)
            self.loss[epoch] = loss_epoch

    def embedding(self, x, layer):
        #         parameter
        # x  n*d numpy array
        # layer 'hidden' 或 'output'，用于标记获得哪一层经过激活函数后的数据

        net1 = np.dot(x, self.w1)

        y = np.tanh(net1 + self.b1)  # batch_size*h

        net2 = np.dot(y, self.w2)

        z = self.sigmoid(net2 + self.b2)

        if (layer == 'hidden'):
            return y

        if (layer == 'output'):
            return z

    def predict(self, x):
        #         parameter
        # x  n*p numpy array

        z = self.embedding(x, 'output')

        return np.argmax(z, axis=1)

    def acc(self, x, y):
        #         parameter
        # x  n*p numpy array
        # y  经过one-hot编码的标签，n*c numpy array

        y_p = self.predict(x)

        y = np.argmax(y, axis=1)

        return np.sum((y == y_p).astype(np.int)) * 1.0 / y.shape[0]