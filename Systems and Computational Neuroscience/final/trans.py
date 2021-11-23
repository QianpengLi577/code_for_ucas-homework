# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 15:35
# @Author  : Qianpeng Li
# @FileName: trans.py
# @Contact : liqianpeng2021@ia.ac.cn

def tarns(x, I, F):
    # x 输入数据  //a number
    # I 整数部分  包括符号位
    # F 小数部分
    dec = int(x * 2 ** F)
    str = '{:016b}'.format(int(dec + 2 ** (I + F)))
    out = '{:04x}'.format(int(str[len(str) - (I + F):len(str)], 2))
    return str[len(str) - (I + F):len(str)], out


w = [0.25, 0.02, 0.004, 0.02, 0.0453, 0.0543, 0.00821, -0.0123, 0.032]
with open("w.txt", "w") as f:
    for w_ in w:
        str, out = tarns(w_, 3, 13)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')
b = [0.12]
with open("b.txt", "w") as f:
    for b_ in b:
        str, out = tarns(w_, 3, 13)
        print('二进制数：', str)
        print('十六进制：', out)
        f.write(str + '\n')
import numpy as np

x = np.random.randint(0, 2, size=(10, 10))
with open("x.txt", "w") as f:
    for i in range(10):
        for j in range(10):
            str, out = tarns(x[i,j], 1, 2)
            print('二进制数：', str)
            print('十六进制：', out)
            f.write(str + '\n')
