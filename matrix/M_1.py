from scipy.linalg import lu
import sys
from pa_lu import *

# numpy                              1.20.2
# scipy                              1.7.1
######################################################
######################################################
# 给出了两个A矩阵的实例，直接运行本文件
# 先输出PA、LU、LL1、UU1、AA1、PP1
# 然后输出scipy实现的A=PLU和本例程实现的PA=LU进行对比
# 对于scipy A=plu，将其改写成p.TA=lu进行展示
# 对于分解结果的展示，上方矩阵为scipy实现的，下方矩阵为本例实现的


A = np.array([[1, 2, 4, 17],
              [3, 6, -12, 3],
              [2, 3, -3, 2],
              [0, 2, -2, 6]])

# A = np.array([[4, 2, 1, 5],
#               [8, 7, 2, 10],
#               [4, 8, 3, 6],
#               [6, 8, 4, 9]])

# A = np.array([[1, 2, -3, 4],
#               [4, 8, 12, -8],
#               [2, 3, 2, 1],
#               [-3, -1, 1, -4]])

A = A.astype(float)

if (A.shape[0] != A.shape[1]):
    print('A非方阵')
    sys.exit()
if (np.linalg.det(A) == 0):
    print('A为奇异阵')
    sys.exit()

A_tp = A.copy()

P, L, U = getp_l_u(A_tp)

A_1=getinv_A(L,U,P)
# print('P:\n', P)
# print('L:\n', L)
# print('U:\n', U)

L1 = getinv_L(L)

U1 = getinv_U(U)

P1 = P.T

A1 = U1.dot(L1.dot(P))

np.set_printoptions(precision=6, suppress=True)

# print('PA=\n', P.dot(A))
# print('LU=\n', L.dot(U))
# print('LL1=\n', L.dot(L1))
# print('UU1=\n', U.dot(U1))
# print('PP1=\n', P.dot(P1))
print('通过LU回代得到的A逆矩阵,验证其为A逆矩阵')
print('AA_1=\n', A_1.dot(A))#通过LU回代得到的A逆矩阵
print('通过L逆、U逆、P逆得到的A逆矩阵,验证其为A逆矩阵')
print('AA1=\n', A.dot(A1))#通过L逆、U逆、P逆得到的A逆矩阵

p, l, u = lu(a=A, permute_l=False)
print('scipy   my_work\n', 'p:\n', p.T, '\n', '\n', P, '\n', 'l:\n', l, '\n', '\n', L, '\n', 'u:\n', u, '\n', '\n', U,
      '\n', 'A1:\n', np.linalg.inv(A), '\n', '\n', A1,'\n','\n', A_1)
