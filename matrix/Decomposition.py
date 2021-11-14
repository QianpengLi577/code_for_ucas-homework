# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 8:57
# @Author  : Qianpeng Li
# @FileName: Decomposition.py
# @Contact : liqianpeng2021@ia.ac.cn

import numpy as np


def getp_l_u(A_tp):
    # 参数
    # A_tp  待分解矩阵

    N = A_tp.shape[0]
    P_t = np.zeros(N)
    L = np.zeros((N, N))
    P = np.zeros((N, N))
    for i in range(N):
        P_t[i] = i  # index 设置初值
    for i in range(N - 1):

        index_max = np.argmax(abs(A_tp[i:N, i])) + i  # 找到当前主元index

        temp_P = P_t[index_max].copy()
        P_t[index_max] = P_t[i]
        P_t[i] = temp_P  # index 交换

        temp_A = A_tp[index_max, :].copy()
        A_tp[index_max, :] = A_tp[i, :]
        A_tp[i, :] = temp_A  # A矩阵两行交换

        temp_L = L[index_max, :].copy()
        L[index_max, :] = L[i, :]
        L[i, :] = temp_L  # L矩阵两行交换

        for j in range(N - i - 1):
            L[j + 1 + i, i] = A_tp[j + 1 + i, i] / A_tp[i, i]  # 得到L参数
            A_tp[j + 1 + i, :] = A_tp[j + 1 + i, :] - A_tp[j + 1 + i, i] / A_tp[i, i] * A_tp[i, :]  # 高斯消元

    for i in range(N):
        P[i, int(P_t[i])] = 1  # 得到P矩阵
    L = L + np.eye(N)  # 得到L矩阵
    U = A_tp.copy()  # U矩阵即消元后的A矩阵
    return (P, L, U)


def getinv_L(L):
    # 参数
    # L 下三角矩阵

    N = L.shape[0]
    L1 = np.eye(N)
    for i in range(N - 1):
        for j in range(i + 1):
            L1[i + 1, j] = -np.dot(L[i + 1, j:i + 1], L1[j:i + 1, j])  # 根据矩阵乘法展开，迭代得到L逆矩阵的值
    return L1


def getinv_U(U):
    # 参数
    # U 上三角矩阵

    N = U.shape[0]
    dg = np.zeros((N, N))
    for i in range(N):
        dg[i, i] = 1.0 / U[i, i]  # U=P.dot(U_L) U1=U_L1.dot(dg) dg=P1,其对角元素为U对角元素的倒数
    U_L_T = U.copy()
    for i in range(N):
        U_L_T[i, :] = U_L_T[i, :] / U[i, i]  # 这里将U_L转置，方便使用L求逆函数，
    return np.transpose(getinv_L(U_L_T.T)).dot(dg)


def getinv_A(L, U, P):
    # 参数
    # L U P 满足 PA=LU

    N = L.shape[0]
    A_1 = np.zeros((N, N))

    for l in range(N):
        y = np.zeros(N)
        e = P[:, l]
        y[0] = e[0] / L[0, 0]
        for i in range(N - 1):
            y[i + 1] = (e[i + 1] - y[0:i + 1].dot(L[i + 1, 0:i + 1])) / L[i + 1, i + 1]

        A_1[N - 1, l] = y[N - 1] / U[N - 1, N - 1]
        for j in range(N - 1):
            A_1[N - 1 - 1 - j, l] = (y[N - 1 - 1 - j] - A_1[N - 1 - j:N, l].dot(U[N - 1 - 1 - j, N - 1 - j:N])) / U[
                N - 2 - j, N - 2 - j]
    # 回代法求解A_1
    return A_1


def norm_2(x):
    # 参数
    # x n*1 np数组
    return np.sqrt(x.T.dot(x))


def Gram_Schmidt(A):
    # 参数
    # A A=QR  Q^{T}Q=I R上三角阵
    # A m*n Q m*n R n*n
    M = A.shape[0]
    N = A.shape[1]
    Q = np.zeros((M, N))
    R = np.zeros((N, N))
    R[0, 0] = norm_2(A[:, 0])
    Q[:, 0] = A[:, 0] / norm_2(A[:, 0])
    for j in range(N - 1):
        R[0, j + 1] = Q[:, 0].T.dot(A[:, j + 1])
    for i in range(N - 1):
        temp = np.zeros(M)
        for m in range(i + 1):
            temp = temp + R[m, i + 1] * Q[:, m]
        q = A[:, i + 1] - temp
        R[i + 1, i + 1] = norm_2(q)
        Q[:, i + 1] = q / norm_2(q)
        for j in range(N - 1 - i - 1):
            R[i + 1, i + 1 + j + 1] = Q[:, i + 1].T.dot(A[:, i + 1 + j + 1])
    return (Q, R)


def Householder_reduction(A):
    # 参数
    # A A=QR
    # A n*n Q n*n R n*n
    N = A.shape[0]
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    P_history = []
    P = np.zeros((N, N))
    for i in range(N - 1):
        x = A_TP[i:N, i]
        e = np.zeros(len(x))
        e[0] = 1
        u = x - 1 * norm_2(x) * e
        u = u.reshape((len(u), 1))
        p = np.eye(len(x)) - 2 * u.dot(u.T) / (u.T.dot(u))
        if (i == 0):
            P = p
        else:
            a = np.eye(N)
            a[i:N, i:N] = p
            P = a.dot(P)
        A_TP[i:N, i:N] = p.dot(A_TP[i:N, i:N])
        P_history.append(p)
    return (P.T, A_TP)  ## A=QR


def Givens_reduction(A):
    # 参数
    # A A=QR givens 分解
    # A n*n Q n*n R n*n
    N = A.shape[0]
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    for i in range(N - 1):
        for j in range(N - 1 - i):
            p_tp = np.eye(N)
            a = A_TP[i, i]
            b = A_TP[i + j + 1, i]
            p_tp[i, i] = a / np.sqrt(a * a + b * b)
            p_tp[i, j + i + 1] = b / np.sqrt(a * a + b * b)
            p_tp[j + i + 1, j + i + 1] = a / np.sqrt(a * a + b * b)
            p_tp[j + i + 1, i] = -b / np.sqrt(a * a + b * b)
            if ((i == 0) & (j == 0)):
                P = p_tp
            else:
                P = p_tp.dot(P)
            A_TP = p_tp.dot(A_TP)
    return (P.T, A_TP)


def Gaussian_Elimination(A):
    # 高斯主元消去法
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    M = A_TP.shape[0]
    N = A_TP.shape[1]
    for i in range(min(M, N)):
        if (i < min(M, N) - 1):
            index_max = abs(A_TP[i:M, i]).argmax() + i
        else:
            index_max = min(M, N) - 1
        temp = A_TP[index_max, :].copy()
        A_TP[index_max, :] = A_TP[i, :]
        A_TP[i, :] = temp
        for j in range(M):
            alpha = A_TP[j, i] / A_TP[i, i]
            if (j != i):
                A_TP[j, :] = A_TP[j, :] - alpha * A_TP[i, :]
    return A_TP


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def find_RA(A):
    GE = Gaussian_Elimination(A)
    # print(GE)
    index = np.where(first_nonzero(GE, axis=1, invalid_val=-1) != -1)
    out = A[:, np.array(index)].reshape(GE[:, np.array(index)].shape[0], GE[:, np.array(index)].shape[2])
    Q, R = Gram_Schmidt(out)
    return Q


def find_NA(A, r):
    # 参数
    # r A的秩
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    M = A_TP.shape[0]
    N = A_TP.shape[1]
    if (N == r):
        return None
    else:
        GE = Gaussian_Elimination(A_TP)
        temp = GE[0:r, :]
        index = np.where(first_nonzero(GE, axis=1, invalid_val=-1) != -1)[0]
        index_all = np.arange(N)
        for i in range(len(index)):
            index_all[index[i]] = -1
        index_free = np.where(index_all != -1)[0]
        x = np.zeros((N, N - r))
        for i in range(len(index_free)):
            x[index_free[i], i] = 1
            for j in range(len(index)):
                x[index[j], i] = -temp[j, index_free[i]] / temp[j, index[j]]
        Q, R = Gram_Schmidt(x)
        return Q


def URV(A):
    # 参数
    # A  A=URV_T  U V正交矩阵
    # A m*n u m*m r m*n v n*n
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    M = A_TP.shape[0]
    N = A_TP.shape[1]
    U = np.zeros((M, M))
    R = np.zeros((M, N))
    V = np.zeros((N, N))

    RA = find_RA(A_TP)
    rank = RA.shape[1]
    NAT = find_NA(A.T, rank)
    RAT = find_RA(A.T)
    NA = find_NA(A, rank)
    U[:, 0:rank] = RA
    U[:, rank:M] = NAT
    V[:, 0:rank] = RAT
    V[:, rank:N] = NA
    R = np.dot(U.T, A.dot(V))
    return (U, R, V)
