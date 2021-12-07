# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 8:57
# @Author  : Qianpeng Li
# @FileName: Decomposition.py
# @Contact : liqianpeng2021@ia.ac.cn

import numpy as np


###########################
###########################
# 样例中  基本上涵盖了所有矩阵的情况
# A 4*4 可逆矩阵
# A_URV 4*3 列满秩矩阵
# A_bad 4*3 rank=1 矩阵

# A = np.array([[1, 2, 4, 17],
#               [3, 6, -12, 3],
#               [2, 3, -3, 2],
#               [0, 2, -2, 6]])
# A_URV = np.array([[1, 2, 4],
#                   [3, 6, -12],
#                   [2, 3, -3],
#                   [0, 2, -2]])
# A_bad=np.array([[1,2,3],[2,4,6],[3,6,9],[4,8,12]])

def can_decomposition(A):
    # 判断是否可以进行分解
    # 对于PA=LU，A要可逆
    # 对于household、givens、施密特正交化需要列满秩
    # 因此直接求是不是列满秩，是的话return true  否则 return false
    # URV分解没什么限制条件

    # 输入参数
    # A m*n numpy array

    # 输出参数
    # rank==col(A)  布尔变量  列满秩？
    # M = A.shape[0]

    # eg:
    # can_decomposition(A)
    # Out[3]: True
    # can_decomposition(A_URV)
    # Out[4]: True
    # can_decomposition(A_bad)
    # Out[5]: False

    N = A.shape[1]
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    GE = Gaussian_Elimination(A_TP)
    # print(GE)
    index = np.where(first_nonzero(GE, axis=1, invalid_val=-1) != -1)[0]
    rank = len(index)
    return rank == N


def getp_l_u(A_tp):
    # 功能
    # PA=LU分解

    # 输入参数
    # A_tp  待分解矩阵  shape:n*n numpy array

    # 输出参数
    # P L U 满足 PA=LU P为旋转矩阵  L为下三角矩阵  U为上三角矩阵  shape均为：n*n numpy array

    # eg
    # A_copy = A.copy()
    # P, L, U = getp_l_u(A_copy)
    # print(P, '\n', L, '\n', U)

    # [[0. 1. 0. 0.]
    #  [0. 0. 0. 1.]
    #  [1. 0. 0. 0.]
    #  [0. 0. 1. 0.]]

    # [[ 1.          0.          0.          0.        ]
    #  [ 0.          1.          0.          0.        ]
    #  [ 0.33333333  0.          1.          0.        ]
    #  [ 0.66666667 -0.5         0.5         1.        ]]

    # [[  3   6 -12   3]
    #  [  0   2  -2   6]
    #  [  0   0   8  16]
    #  [  0   0   0  -5]]

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
    # 功能
    # 计算L的逆

    # 输入参数
    # L 下三角矩阵  n*n numpy array

    # 输出参数
    # L1 L矩阵的逆  n*n numpy array

    N = L.shape[0]
    L1 = np.eye(N)
    for i in range(N - 1):
        for j in range(i + 1):
            L1[i + 1, j] = -np.dot(L[i + 1, j:i + 1], L1[j:i + 1, j])  # 根据矩阵乘法展开，迭代得到L逆矩阵的值
    return L1


def getinv_U(U):
    # 功能
    # 计算U的逆

    # 输入参数
    # U 上三角矩阵  n*n numpy array

    # 输出参数
    # U1 U的逆矩阵  n*n numpy array

    N = U.shape[0]
    dg = np.zeros((N, N))
    for i in range(N):
        dg[i, i] = 1.0 / U[i, i]  # U=P.dot(U_L) U1=U_L1.dot(dg) dg=P1,其对角元素为U对角元素的倒数
    U_L_T = U.copy()
    for i in range(N):
        U_L_T[i, :] = U_L_T[i, :] / U[i, i]  # 这里将U_L转置，方便使用L求逆函数，
    return np.transpose(getinv_L(U_L_T.T)).dot(dg)


def getinv_A(L, U, P):
    # 功能
    # 通过PA=LU 得到A的逆

    # 输入参数
    # L U P 满足 PA=LU  均为 n*n numpy array

    # 输出参数
    # A1  A矩阵的逆  n*n numpy array

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
    # 功能
    # 计算向量二范数

    # 输入参数
    # x n*1  numpy array

    # 输出
    # \sqrt x^{T} \cdot x

    # eg
    # x=np.array([1,2,3])
    # norm_2(x)
    # Out[11]: 3.7416573867739413

    return np.sqrt(x.T.dot(x))


def Gram_Schmidt(A):
    # 功能
    # 施密特正交化

    # 输入参数
    # A m*n numpy array m>=n A=QR  Q^{T}Q=I R上三角阵

    # 输出参数
    #  Q m*n R n*n  numpy array  Q^{T}Q=I R上三角阵

    # eg
    # Q,R=Gram_Schmidt(A)
    # print(Q,'\n',R)

    # [[ 0.26726124  0.06579517  0.88396     0.37796447]
    #  [ 0.80178373  0.19738551 -0.41871789  0.37796447]
    #  [ 0.53452248 -0.32897585  0.18609684 -0.75592895]
    #  [ 0.          0.92113237  0.09304842 -0.37796447]]
    #
    # [[  3.74165739   6.94879229 -10.15592719   8.01783726]
    #  [  0.           2.17124059  -2.96078263   6.57951695]
    #  [  0.           0.           7.81606737  14.70165052]
    #  [  0.           0.           0.           3.77964473]]

    # Q,R=Gram_Schmidt(A_URV)
    # print(Q,'\n',R)

    # [[ 0.26726124  0.06579517  0.88396   ]
    #  [ 0.80178373  0.19738551 -0.41871789]
    #  [ 0.53452248 -0.32897585  0.18609684]
    #  [ 0.          0.92113237  0.09304842]]
    #
    # [[  3.74165739   6.94879229 -10.15592719]
    #  [  0.           2.17124059  -2.96078263]
    #  [  0.           0.           7.81606737]]

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
    # 功能
    # household 约减

    # 输入参数
    # A m*n numpy array(m>=n)

    # 输出
    # Q m*m R m*n numpy array
    # Q^{T}Q=I R上三角阵

    # eg
    # Q,R=Householder_reduction(A)
    # print(Q,'\n',R)

    # [[ 0.267261  0.065795  0.88396  -0.377964]
    #  [ 0.801784  0.197386 -0.418718 -0.377964]
    #  [ 0.534522 -0.328976  0.186097  0.755929]
    #  [ 0.        0.921132  0.093048  0.377964]]

    #  [[  3.741657   6.948792 -10.155927   8.017837]
    #  [  0.         2.171241  -2.960783   6.579517]
    #  [  0.         0.         7.816067  14.701651]
    #  [  0.         0.         0.        -3.779645]]

    # Q,R=Householder_reduction(A_URV)
    # print(Q,'\n',R)
    #
    # [[ 0.267261  0.065795  0.88396  -0.377964]
    #  [ 0.801784  0.197386 -0.418718 -0.377964]
    #  [ 0.534522 -0.328976  0.186097  0.755929]
    #  [ 0.        0.921132  0.093048  0.377964]]

    # [[  3.741657   6.948792 -10.155927]
    #  [  0.         2.171241  -2.960783]
    #  [  0.         0.         7.816067]
    #  [  0.         0.         0.      ]]

    M = A.shape[0]
    N = A.shape[1]
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    P_history = []
    P = np.zeros((M, M))
    for i in range(M - 1):
        x = A_TP[i:M, i]
        e = np.zeros(len(x))
        e[0] = 1
        u = x - 1 * norm_2(x) * e
        u = u.reshape((len(u), 1))
        p = np.eye(len(x)) - 2 * u.dot(u.T) / (u.T.dot(u))
        if (i == 0):
            P = p
        else:
            a = np.eye(M)
            a[i:M, i:M] = p
            P = a.dot(P)
        A_TP[i:M, i:N] = p.dot(A_TP[i:M, i:N])
        P_history.append(p)
    return (P.T, A_TP)  ## A=QR


def Givens_reduction(A):
    # 功能
    # Givens 约减

    # 输入参数
    # A m*n (m>=n) numpy array

    # 输出
    # Q m*m R m*n numpy array
    # Q^{T}Q=I R上三角阵

    # eg
    # Q,R=Givens_reduction(A)
    # print(Q,'\n',R)

    # [[ 0.267261  0.065795  0.88396   0.377964]
    #  [ 0.801784  0.197386 -0.418718  0.377964]
    #  [ 0.534522 -0.328976  0.186097 -0.755929]
    #  [ 0.        0.921132  0.093048 -0.377964]]

    #  [[  3.741657   6.948792 -10.155927   8.017837]
    #  [  0.         2.171241  -2.960783   6.579517]
    #  [ -0.        -0.         7.816067  14.701651]
    #  [ -0.        -0.        -0.         3.779645]]

    # Q,R=Givens_reduction(A_URV)
    # print(Q,'\n',R)

    # [[ 0.267261  0.065795  0.88396   0.377964]
    #  [ 0.801784  0.197386 -0.418718  0.377964]
    #  [ 0.534522 -0.328976  0.186097 -0.755929]
    #  [ 0.        0.921132  0.093048 -0.377964]]

    # [[  3.741657   6.948792 -10.155927]
    #  [  0.         2.171241  -2.960783]
    #  [ -0.        -0.         7.816067]
    #  [ -0.        -0.        -0.      ]]

    M = A.shape[0]
    N = A.shape[1]
    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    for i in range(N):
        for j in range(M - 1 - i):
            p_tp = np.eye(M)
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
    # 功能
    # 高斯主元消去法(主元上方也消去)

    # 输入参数
    # A 输入矩阵 m*n numpy array

    # 输出参数
    # A_TP 经过高斯消去的矩阵

    # eg
    # print(Gaussian_Elimination(A))
    # [[ 3.  0.  0.  0.]
    #  [ 0.  2.  0.  0.]
    #  [ 0.  0.  8.  0.]
    #  [ 0.  0.  0. -5.]]

    # print(Gaussian_Elimination(A_bad))
    # [[ 4.  8. 12.]
    #  [ 0.  0.  0.]
    #  [ 0.  0.  0.]
    #  [ 0.  0.  0.]]

    A_TP = A.copy()
    A_TP = A_TP.astype(float)
    M = A_TP.shape[0]
    N = A_TP.shape[1]
    for i in range(min(M, N)):
        if (i < min(M, N) - 1):
            if (abs(A_TP[i:M, i]).max() == 0): return A_TP
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
    # 功能
    # 寻找每行第一个不为零元素的纵坐标

    # 输入参数
    # arr m*n 矩阵
    # axis 按行寻找还是按列寻找
    # invalid_val  对于全零行，找不到index，将其index输出为invalid_val

    # 输出参数
    # 主元所在的列

    # eg
    # rint(first_nonzero(Gaussian_Elimination(A),axis=1,invalid_val=-1))
    # [0 1 2 3]
    # print(first_nonzero(Gaussian_Elimination(A_URV),axis=1,invalid_val=-1))
    # [ 0  1  2 -1]
    # print(first_nonzero(Gaussian_Elimination(A_bad),axis=1,invalid_val=-1))
    # [ 0 -1 -1 -1]

    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def find_RA(A):
    # 功能
    # 寻找A的值域空间

    # 输入参数
    # A m*n numpu array

    # 输出参数
    # Q m*k  numpy array
    # Q^{T}Q=I  即为A值域空间的一组标准正交基
    # 通过施密特正交化得到

    # eg
    # print(find_RA(A))
    # [[ 0.267261  0.065795  0.88396   0.377964]
    #  [ 0.801784  0.197386 -0.418718  0.377964]
    #  [ 0.534522 -0.328976  0.186097 -0.755929]
    #  [ 0.        0.921132  0.093048 -0.377964]]

    # print(find_RA(A_bad))
    # [[0.182574]
    #  [0.365148]
    #  [0.547723]
    #  [0.730297]]

    GE = Gaussian_Elimination(A)
    # print(GE)
    index = np.where(first_nonzero(GE, axis=1, invalid_val=-1) != -1)
    out = A[:, np.array(index)].reshape(GE[:, np.array(index)].shape[0], GE[:, np.array(index)].shape[2])
    Q, R = Gram_Schmidt(out)
    return Q


def find_NA(A, r):
    # 功能
    # 寻找A的零空间

    # 输入参数
    # r A的秩
    # A m*n numpy array

    # 输出参数
    # Q n*k numpy array
    # Q^{T}Q=I  即为A零空间的一组标准正交基
    # 通过施密特正交化得到

    # eg
    # print(find_NA(A,4)) # r(A)=4
    # None
    # print(find_NA(A_bad,1)) # r(A_bad)=1
    # [[-0.894427 -0.358569]
    #  [ 0.447214 -0.717137]
    #  [ 0.        0.597614]]

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
    # 功能
    # URV分解

    # 输入参数
    # A  m*n numpy array

    # 输出参数
    # u m*m r m*n v n*n numpy array
    # 满足A=URV_T  U V正交矩阵

    # eg
    # U, R, V = URV(A_URV)
    # print(U,'\n',R,'\n',V)
    #
    # [[ 0.267261  0.065795  0.88396  -0.377964]
    #  [ 0.801784  0.197386 -0.418718 -0.377964]
    #  [ 0.534522 -0.328976  0.186097  0.755929]
    #  [ 0.        0.921132  0.093048  0.377964]]

    #  [[-5.015622 11.84124   0.239046]
    #  [-1.636776  3.139845 -0.971008]
    #  [ 6.822423 -3.81385   0.      ]
    #  [-0.       -0.        0.      ]]

    #  [[ 0.218218  0.39036   0.894427]
    #  [ 0.436436  0.78072  -0.447214]
    #  [ 0.872872 -0.48795   0.      ]]

    # U, R, V = URV(A_bad)
    # print(U,'\n',R,'\n',V)

    # [[ 0.182574 -0.894427 -0.358569 -0.19518 ]
    #  [ 0.365148  0.447214 -0.717137 -0.39036 ]
    #  [ 0.547723  0.        0.597614 -0.58554 ]
    #  [ 0.730297  0.        0.        0.68313 ]]

    #  [[20.493902 -0.        0.      ]
    #  [-0.        0.        0.      ]
    #  [ 0.       -0.        0.      ]
    #  [-0.        0.       -0.      ]]

    #  [[ 0.267261 -0.894427 -0.358569]
    #  [ 0.534522  0.447214 -0.717137]
    #  [ 0.801784  0.        0.597614]]

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
