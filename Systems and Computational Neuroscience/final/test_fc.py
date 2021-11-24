# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 9:35
# @Author  : Qianpeng Li
# @FileName: test_fc.py
# @Contact : liqianpeng2021@ia.ac.cn

def tarns(x, I, F):
    # x 输入数据  //a number
    # I 整数部分  包括符号位
    # F 小数部分
    dec = int(x * 2 ** F)
    str = '{:016b}'.format(int(dec + 2 ** (I + F)))
    out = '{:04x}'.format(int(str[len(str) - (I + F):len(str)], 2))
    return str[len(str) - (I + F):len(str)], out


x = [0.75, 0.25, 0, 0.5, 0.75, 0.25, 1, 1, 0, 1, 0.25, 0.5]
with open("E:/UCAS/x_fc.txt", "w") as f:
    for x_ in x:
        str, out = tarns(x_, 1, 2)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')

import numpy as np
w=np.random.random((12,10))
with open("E:/UCAS/w_fc.txt", "w") as f:
    for j in range(10):
        for i in range(12):
            str, out = tarns(w[i,j], 3, 13)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')

b=np.random.random((10))
with open("E:/UCAS/b_fc.txt", "w") as f:
    for j in range(10):
            str, out = tarns(b[j], 3, 13)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')
y=np.dot(np.array(x).T,w)+b
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
out_fpga=np.zeros(10)
for i in range(10):
    out_fpga[i]=int(result[0][16*(10-i-1):16*(10-i)],2)/(2**13)
print(np.sum(np.abs(y-out_fpga)))
