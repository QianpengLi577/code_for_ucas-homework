# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 15:35
# @Author  : Qianpeng Li
# @FileName: trans.py
# @Contact : liqianpeng2021@ia.ac.cn

#     测试卷积部分

def tarns(x, I, F):
    # x 输入数据  //a number
    # I 整数部分  包括符号位
    # F 小数部分
    dec = int(x * 2 ** F)
    str = '{:016b}'.format(int(dec + 2 ** (I + F)))
    out = '{:04x}'.format(int(str[len(str) - (I + F):len(str)], 2))
    return str[len(str) - (I + F):len(str)], out
def decode(x,I,F,unsigned):
    # x str
    # unsigned True or False
    sum=0
    if (unsigned):
        flag=1
    else:
        flag=-1
    for i in range(len(x)):
        if (i==0):
            sum += flag*int(x[i],2)*2**(I+F-1-i)
        else:
            sum += int(x[i], 2) * 2 ** (I + F - 1 - i)
    return sum/(2**F)

w = [-0.25, 0.02, 0.004, 0.02, 0.0453, 0.0543, 0.00821, -0.0123, 0.032]
with open("E:/UCAS/w.txt", "w") as f:
    for w_ in w:
        str, out = tarns(w_, 3, 13)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')
b = [-0.12]
with open("E:/UCAS/b.txt", "w") as f:
    for b_ in b:
        str, out = tarns(b_, 3, 13)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')
import numpy as np

x = np.random.randint(0, 2, size=(10, 10))
with open("E:/UCAS/x.txt", "w") as f:
    for i in range(10):
        for j in range(10):
            str, out = tarns(x[i,j], 1, 2)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')
w=np.array(w).reshape((3,3))
out=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        out[i,j]=np.sum(x[i:i+3,j:j+3]*w)+b

x_fc = [0.75, 0.25, 0, 0.5, 0.25, 0.25, 0, 0, 0, 1, 0.25, 0.5]
with open("E:/UCAS/x_fc.txt", "w") as f:
    for x_ in x_fc:
        str, out = tarns(x_, 1, 2)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')

import numpy as np
w_fc=np.random.random((12,10))
with open("E:/UCAS/w_fc.txt", "w") as f:
    for j in range(10):
        for i in range(12):
            str, out = tarns(w_fc[i,j], 3, 13)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')

b_fc=np.random.random((10))
with open("E:/UCAS/b_fc.txt", "w") as f:
    for j in range(10):
            str, out = tarns(b_fc[j], 3, 13)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')
y_fc=np.dot(np.array(x_fc).T,w_fc)+b_fc
print('finsh')
name = 'E:/UCAS/result_fc.txt'
# result=np.loadtxt(name,dtype='str',delimiter='\n')
result=[]
with open(name, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        result.append(line)#去掉列表中每一个元素的换行符
        # print(line)
result=np.array(result)
out_fpga_fc=np.zeros(10)
for i in range(10):
    out_fpga_fc[i]=decode(result[0][16*(10-i-1):16*(10-i)],3,13,False)
print(np.sum(np.abs(y_fc-out_fpga_fc-b_fc)))


name = 'E:/UCAS/result.txt'
out=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        out[i,j]=np.sum(x[i:i+3,j:j+3]*w)+b
# result=np.loadtxt(name,dtype='str',delimiter='\n')
result=[]
with open(name, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        result.append(line)#去掉列表中每一个元素的换行符
        # print(line)
result=np.array(result)
out_fpga=np.zeros((8,8))
for i in range(8*8):
    out_fpga[int(i/8),i%8]=decode(result[0][16*(8*8-i-1):16*(8*8-i)],3,13,False)
print(np.sum(np.abs(out-out_fpga)))
