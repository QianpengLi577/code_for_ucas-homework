# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 14:45
# @Author  : Qianpeng Li
# @FileName: Equation_solving.py
# @Contact : liqianpeng2021@ia.ac.cn

import numpy as np


# from Decomposition import Gram_Schmidt,Gaussian_Elimination,first_nonzero,find_NA

# b = np.array([[1], [2], [3], [4]])
# b = b.astype(float)
def pa_lu_equation(P, L, U, b):
    # 功能
    # 通过PA=LU分解求解Ax=b

    # 输入参数
    # P L U S.T. PA=LU  均为n*n numpy array
    # !!!!!!!!!!!   A必须可逆
    # b n*1 numpy array

    # 求解过程
    # LUx=Pb  -->  Ly=Pb  --> Ux=y --> x

    # 输出参数
    # x n*1 numpy array   s.t. Ax=b

    # eg
    #
    # print(pa_lu_equation(P, L, U, b))
    # [[-3.95    ]
    #  [ 5.541667]
    #  [ 1.441667]
    #  [-0.7     ]]

    b1 = np.dot(P, b)
    M = P.shape[0]
    N = P.shape[1]
    y = np.zeros((M, 1))
    x = np.zeros((M, 1))
    for i in range(M):
        if (i == 0):
            y[i] = b1[i] / L[i, i]
        else:
            y[i] = (b1[i] - L[i, 0:i].dot(y[0:i])) / L[i, i]
    for j in range(M):
        if (j == 0):
            x[M - 1 - j] = y[M - 1 - j] / U[M - 1 - j, M - 1 - j]
        else:
            x[M - 1 - j] = (y[M - 1 - j] - U[M - 1 - j, M - j:M].dot(x[M - j:M])) / U[M - 1 - j, M - 1 - j]
    return x


def qr_equation(Q, R, b):
    # 功能
    # 通过QR分解求解AX=b

    # 参数
    # A=QR R上三角矩阵  Q正定矩阵  numpy array
    # b numpy array

    # 过程
    # QRx=b -> Qy=b -> y=Q^{T}b  -> 求解 Ux = y

    # 输出参数
    # x n*1 numpy array   s.t. Ax=b

    # Q,R=Gram_Schmidt(A)
    # print(qr_equation(Q, R, b))
    # [[-3.95    ]
    #  [ 5.541667]
    #  [ 1.441667]
    #  [-0.7     ]]

    M = Q.shape[0]
    N = Q.shape[1]
    y = Q.T.dot(b)
    x = np.zeros((M, 1))
    for i in range(M):
        if (i == 0):
            x[M - 1 - i] = y[M - 1 - i] / R[M - 1 - i, M - 1 - i]
        else:
            x[M - 1 - i] = (y[M - 1 - i] - R[M - 1 - i, M - i:M].dot(x[M - i:M])) / R[M - 1 - i, M - 1 - i]
    return x

# def general_solving_equation(A,b):
