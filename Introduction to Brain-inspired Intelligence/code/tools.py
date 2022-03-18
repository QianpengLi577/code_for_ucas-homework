# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/13 9:43
@Author  : Qianpeng Li
@FileName: tools.py
@Contact : liqianpeng2021@ia.ac.cn
"""
import numpy as np
import torch


# 量化 func
def float_to_fixed_array(input, total_bit, float_bit):
    # x numpy darray
    # total_bit  int number
    # float_bit  int number

    x = input.numpy()

    FL = float_bit  # 用 float_bit bit表示小数部分
    IL = total_bit - FL  # 用 IL bit表示整数部分（1 bit为符号位，IL-1 bit为整数）
    MIN = -(1 << (IL - 1))  # 可表示的最小值
    MAX = -MIN - 2 ** (-FL)  # 可表示的最大值

    x = np.clip(x, MIN, MAX)
    sig = np.ones(x.shape)  # 符号位
    idx = np.where(x < 0)
    sig[idx] = -1
    x = abs(x)
    q = np.trunc(x)
    x -= q  # 此时x为小数部分
    e = 1
    for i in range(FL):
        x *= 2
        e /= 2
        idx_1 = np.where(x >= 1)
        x[idx_1] -= 1
        q[idx_1] += e
    idx_2 = np.where(x >= 0.5)
    q[idx_2] += e
    q *= sig
    q = np.clip(q, MIN, MAX)

    return torch.from_numpy(q)


# tb
# randArray = np.random.random(size=(2, 4))
# print(randArray)
# result = float_to_fixed_array(randArray, total_bit=8, float_bit=2)
# print(result)


# Poisson-coding
def Poisson_coding(x, time_step):
    # x numpy darray
    # time_step int number
    input = x.numpy()
    shape = list(input.shape)
    shape.insert(0, time_step)
    result = np.zeros(tuple(shape))
    for i in range(time_step):
        result[i] = (input > np.random.random(input.shape)).astype(np.float64)
    return torch.from_numpy(result)


# tb
# print('poisson')
# poisson_result = Poisson_coding(randArray, 20)
# print(poisson_result)
# print(poisson_result.shape)


# TTFS time to first spike
def TTFS(x, time_step, max, min):
    fq = (max - min) * 1.0
    input = x.numpy()
    shape = list(input.shape)
    shape.insert(0, time_step)
    result = np.zeros(tuple(shape))
    time = np.around((max - input) / fq * (time_step - 1)).astype(int)
    for i in range(input.shape[0]):
        for j in range(input.shape[2]):
            for m in range(input.shape[3]):
                result[time[i,0,j,m]][i,0,j,m]=1
    return torch.from_numpy(result)

# print('TTFS')
# TTFS_result = TTFS(randArray, 20, max=1.0, min=0.0)
# print(TTFS_result)
# print(TTFS_result.shape)
