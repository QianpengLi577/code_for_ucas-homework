# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 15:41
# @Author  : Qianpeng Li
# @FileName: get_determinant.py
# @Contact : liqianpeng2021@ia.ac.cn

from Decomposition import first_nonzero

# A = np.array([[1, 2, 4, 17],
#               [3, 6, -12, 3],
#               [2, 3, -3, 2],
#               [0, 2, -2, 6]])
def Reverse_order_number(x):
    # 功能
    # 求逆序数

    # 输入
    # x list or n*1 numpy array

    # 输出
    # 逆序数  一个数

    # eg
    # x=[1,2,3,5,4]
    # print(Reverse_order_number(x))
    # 1

    N = len(x)
    sum = 0
    for i in range(N - 1):
        for j in range(N - i - 1):
            if (x[i] > x[j + i + 1]):
                sum += 1
    return sum


def f(x):
    # 功能
    # 输出1 or -1   用于逆序数指定的系数

    # 输入
    # x 一个数

    # 输出
    # 1 或 -1

    if (x == 0):
        return 1
    else:
        return -1


def pa_lu_det(P, L, U):
    # 功能
    # 通过PA=LU获得det(A)

    # 输入参数
    # P L U   S.T. PA=LU  n*n numpy array

    # 输出参数
    # det()  一个数

    # eg
    # A_copy = A.copy()
    # P, L, U = getp_l_u(A_copy)
    # print(pa_lu_det(P,L,U))
    # 240

    N = U.shape[0]
    x = first_nonzero(P, axis=1, invalid_val=-1) + 1
    detp = f(Reverse_order_number(x) % 2)
    detlu = 1
    for i in range(N):
        detlu *= U[i, i]
    return (detp * detlu)


def permutation(nums, p, q, s):
    # 功能
    # 递归生成n!个排列数，用于生成矩阵列坐标index

    # 输入参数
    # nums 列坐标范围 list
    # p q 用于排列的数的范围  为两个实数
    # s list

    # 输出参数
    # 无 n!排列数直接存放在s内

    # eg
    # p=[]
    # nums = [i for i in range(3)]
    # permutation(nums, 0, len(nums), p)
    # print(p)
    # [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]

    if p == q:
        s.append(list(nums))
    else:
        for i in range(p, q):
            nums[i], nums[p] = nums[p], nums[i]
            permutation(nums, p + 1, q, s)
            nums[i], nums[p] = nums[p], nums[i]


def general_det(A):
    # 功能
    # 通用的求det的方法：考虑到尽管正交矩阵的行列式为正1或者－1，但确定符号还是有些困难，
    # 因此求det(Q)所花的时间和直接获得det(A)的时间所差无几

    # 输入参数
    # A 输入矩阵 n*n

    # 输出参数
    # det 一个实数

    # eg
    # print(general_det(A))
    # 240

    N = A.shape[0]
    p = []
    nums = [i for i in range(N)]
    permutation(nums, 0, len(nums), p)
    sum = 0
    for i in range(len(p)):
        temp = 1
        for j in range(len(p[i])):
            temp *= A[j, p[i][j]]
        sum += temp * f(Reverse_order_number(p[i]) % 2)
    return sum
