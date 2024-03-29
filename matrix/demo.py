# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 19:43
# @Author  : Qianpeng Li
# @FileName: demo.py
# @Contact : liqianpeng2021@ia.ac.cn

from Decomposition import *
from Equation_solving import *
from get_determinant import *


def method(A, b, str):
    # 功能
    # 实现分解，求解方程组以及det

    # 输入
    # A 稀疏矩阵 m*n numpy array
    # b 方程右边的矩阵 m*1 numpy array
    # str 一个字符串  通过其选择使用什么方法
    # {'PA_LU','Gram_Schmidt','Householder_reduction','Givens_reduction','URV'}

    # 输出
    # 根据str的选择，所执行的结果直接print

    A_copy = A.copy()
    if (~can_decomposition(A_copy)&(str!='URV')):
        print('A不是列满秩的，无法进行'+str+'分解')
    elif (str == 'PA_LU'):
        if (A_copy.shape[0]!=A_copy.shape[1]):
            print('A不是方阵')
        else :
            print('method: ', str)
            P, L, U = getp_l_u(A_copy)
            print('P')
            print(P)
            print('L')
            print(L)
            print('U')
            print(U)
            print('验证PA=LU')
            print(np.dot(P, A), '=\n', np.dot(L, U))
            x = pa_lu_equation(P, L, U, b)
            print('x')
            print(x)
            print('验证Ax=b')
            print(A.dot(x))
            detA = pa_lu_det(P, L, U)
            print('detA=', detA)

    elif (str == 'Gram_Schmidt'):
        print('method: ', str)
        Q, R = Gram_Schmidt(A)
        print('Q')
        print(Q)
        print('R')
        print(R)
        print('验证QR=A')
        print(np.dot(Q, R), '=\n', A)
        x = qr_equation(Q, R, b)
        print('x')
        print(x)
        print('验证Ax=b')
        print(A.dot(x))
        detA = general_det(A)
        print('detA=', detA)

    elif (str == 'Householder_reduction'):
        print('method: ', str)
        Q, R = Householder_reduction(A)
        print('Q')
        print(Q)
        print('R')
        print(R)
        print('验证QR=A')
        print(np.dot(Q, R), '=\n', A)
        x = qr_equation(Q, R, b)
        print('x')
        print(x)
        print('验证Ax=b')
        print(A.dot(x))
        detA = general_det(A)
        print('detA=', detA)

    elif (str == 'Givens_reduction'):
        print('method: ', str)
        Q, R = Givens_reduction(A)
        print('Q')
        print(Q)
        print('R')
        print(R)
        print('验证QR=A')
        print(np.dot(Q, R), '=\n', A)
        x = qr_equation(Q, R, b)
        print('x')
        print(x)
        print('验证Ax=b')
        print(A.dot(x))
        detA = general_det(A)
        print('detA=', detA)

    elif (str == 'URV'):
        # 当用URV分解的时候，考虑到R矩阵不便于求解方程组
        # 同时为了显示R矩阵的特性，所以不进行行列式的求解以及Ax=b的求解
        print('method: ', str)
        U, R, V = URV(A)
        print('U')
        print(U)
        print('R')
        print(R)
        print('V')
        print(V)
        print('验证URV^{T}=A')
        print(np.dot(U, R.dot(V.T)), '=\n', A)
    else:
        print('关键词错误')


A = np.array([[1, 2, 4, 17],
              [3, 6, -12, 3],
              [2, 3, -3, 2],
              [0, 2, -2, 6]])
A = A.astype(float)
b = np.array([[1], [2], [3], [4]])
b = b.astype(float)
A_URV = np.array([[1, 2, 4],
                  [3, 6, -12],
                  [2, 3, -3],
                  [0, 2, -2]])
A_URV = A_URV.astype(float)

A_bad=np.array([[1,2,3],[2,4,6],[3,6,9],[4,8,12]])
A_bad=A_bad.astype(float)


#  输入矩阵A  b请一定设置为numpy array   type：float
np.set_printoptions(precision=6, suppress=True)
A_tp = A.copy()

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
method(A_tp, b, 'PA_LU')

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
method(A_tp, b, 'Gram_Schmidt')

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
method(A_tp, b, 'Householder_reduction')

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
method(A_tp, b, 'Givens_reduction')

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
method(A_URV, b, 'URV')

print('**************************************************************')
print('**************************************************************')
print('**************************************************************')
print('以下是异常情况：非列满秩矩阵、非方阵')
method(A_URV, b, 'PA_LU')
method(A_bad, b, 'PA_LU')
method(A_bad, b, 'Gram_Schmidt')
method(A_bad, b, 'Householder_reduction')
method(A_bad, b, 'Givens_reduction')

#           运行上述程序应有如下结果


# **************************************************************
# **************************************************************
# **************************************************************
# method:  PA_LU
# P
# [[0. 1. 0. 0.]
#  [0. 0. 0. 1.]
#  [1. 0. 0. 0.]
#  [0. 0. 1. 0.]]
# L
# [[ 1.        0.        0.        0.      ]
#  [ 0.        1.        0.        0.      ]
#  [ 0.333333  0.        1.        0.      ]
#  [ 0.666667 -0.5       0.5       1.      ]]
# U
# [[  3.   6. -12.   3.]
#  [  0.   2.  -2.   6.]
#  [  0.   0.   8.  16.]
#  [  0.   0.   0.  -5.]]
# 验证PA=LU
# [[  3.   6. -12.   3.]
#  [  0.   2.  -2.   6.]
#  [  1.   2.   4.  17.]
#  [  2.   3.  -3.   2.]] =
#  [[  3.   6. -12.   3.]
#  [  0.   2.  -2.   6.]
#  [  1.   2.   4.  17.]
#  [  2.   3.  -3.   2.]]
# x
# [[-3.95    ]
#  [ 5.541667]
#  [ 1.441667]
#  [-0.7     ]]
# 验证Ax=b
# [[1.]
#  [2.]
#  [3.]
#  [4.]]
# detA= 240.0
# **************************************************************
# **************************************************************
# **************************************************************
# method:  Gram_Schmidt
# Q
# [[ 0.267261  0.065795  0.88396   0.377964]
#  [ 0.801784  0.197386 -0.418718  0.377964]
#  [ 0.534522 -0.328976  0.186097 -0.755929]
#  [ 0.        0.921132  0.093048 -0.377964]]
# R
# [[  3.741657   6.948792 -10.155927   8.017837]
#  [  0.         2.171241  -2.960783   6.579517]
#  [  0.         0.         7.816067  14.701651]
#  [  0.         0.         0.         3.779645]]
# 验证QR=A
# [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]] =
#  [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]]
# x
# [[-3.95    ]
#  [ 5.541667]
#  [ 1.441667]
#  [-0.7     ]]
# 验证Ax=b
# [[1.]
#  [2.]
#  [3.]
#  [4.]]
# detA= 240.0
# **************************************************************
# **************************************************************
# **************************************************************
# method:  Householder_reduction
# Q
# [[ 0.267261  0.065795  0.88396  -0.377964]
#  [ 0.801784  0.197386 -0.418718 -0.377964]
#  [ 0.534522 -0.328976  0.186097  0.755929]
#  [ 0.        0.921132  0.093048  0.377964]]
# R
# [[  3.741657   6.948792 -10.155927   8.017837]
#  [  0.         2.171241  -2.960783   6.579517]
#  [  0.         0.         7.816067  14.701651]
#  [  0.         0.         0.        -3.779645]]
# 验证QR=A
# [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]] =
#  [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]]
# x
# [[-3.95    ]
#  [ 5.541667]
#  [ 1.441667]
#  [-0.7     ]]
# 验证Ax=b
# [[1.]
#  [2.]
#  [3.]
#  [4.]]
# detA= 240.0
# **************************************************************
# **************************************************************
# **************************************************************
# method:  Givens_reduction
# Q
# [[ 0.267261  0.065795  0.88396   0.377964]
#  [ 0.801784  0.197386 -0.418718  0.377964]
#  [ 0.534522 -0.328976  0.186097 -0.755929]
#  [ 0.        0.921132  0.093048 -0.377964]]
# R
# [[  3.741657   6.948792 -10.155927   8.017837]
#  [  0.         2.171241  -2.960783   6.579517]
#  [ -0.        -0.         7.816067  14.701651]
#  [ -0.        -0.        -0.         3.779645]]
# 验证QR=A
# [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]] =
#  [[  1.   2.   4.  17.]
#  [  3.   6. -12.   3.]
#  [  2.   3.  -3.   2.]
#  [  0.   2.  -2.   6.]]
# x
# [[-3.95    ]
#  [ 5.541667]
#  [ 1.441667]
#  [-0.7     ]]
# 验证Ax=b
# [[1.]
#  [2.]
#  [3.]
#  [4.]]
# detA= 240.0
# **************************************************************
# **************************************************************
# **************************************************************
# method:  URV
# U
# [[ 0.267261  0.065795  0.88396  -0.377964]
#  [ 0.801784  0.197386 -0.418718 -0.377964]
#  [ 0.534522 -0.328976  0.186097  0.755929]
#  [ 0.        0.921132  0.093048  0.377964]]
# R
# [[-5.015622 11.84124   0.239046]
#  [-1.636776  3.139845 -0.971008]
#  [ 6.822423 -3.81385   0.      ]
#  [-0.       -0.        0.      ]]
# V
# [[ 0.218218  0.39036   0.894427]
#  [ 0.436436  0.78072  -0.447214]
#  [ 0.872872 -0.48795   0.      ]]
# 验证URV^{T}=A
# [[  1.   2.   4.]
#  [  3.   6. -12.]
#  [  2.   3.  -3.]
#  [ -0.   2.  -2.]] =
#  [[  1.   2.   4.]
#  [  3.   6. -12.]
#  [  2.   3.  -3.]
#  [  0.   2.  -2.]]
# **************************************************************
# **************************************************************
# **************************************************************
# 以下是异常情况：非列满秩矩阵、非方阵
# A不是方阵
# A不是列满秩的，无法进行PA_LU分解
# A不是列满秩的，无法进行Gram_Schmidt分解
# A不是列满秩的，无法进行Householder_reduction分解
# A不是列满秩的，无法进行Givens_reduction分解