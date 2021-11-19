# -*- coding: utf-8 -*-
# @Time    : 2021/11/13 18:10
# @Author  : Qianpeng Li
# @FileName: simulation.py
# @Contact : liqianpeng2021@ia.ac.cn
import matplotlib.pyplot as plt
import numpy as np

from neurons import LIF, HH, Izh

###############################
############# L I F ###########

T = 300  # 仿真时长  300ts
dt = 0.01  # 仿真步长  0.01ms
I = np.ones(int(T / dt))
I[0:int(6 / dt)] = 0  # 构造电流
node1 = LIF(1.0, 20.0, 0, 12.0, 1.3)
v = node1.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt), v)
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('LIF')
plt.savefig('LIF.png')
plt.show()

##################################
############# Izh ################

T = 3000
I = 4 * np.ones(int(T / dt))
node2 = Izh(0.02, 0.2, -65, 8)
v = node2.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-RS')
plt.savefig('Izh-RS.png')
plt.show()

I = 4 * np.ones(int(T / dt))
node3 = Izh(0.02, 0.2, -50, 2)
v = node3.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-CH')
plt.savefig('Izh-CH.png')
plt.show()

I = 4 * np.ones(int(T / dt))
node4 = Izh(0.1, 0.2, -65, 8)
v = node4.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-FS')
plt.savefig('Izh-FS.png')
plt.show()

#########################
########## HH ###########
I = 2 * np.ones(int(T / dt))
node5 = HH(120, 36, 0.3)
v = node5.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 40)], v[0:int(T / dt / 40)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('HH')
plt.savefig('HH.png')
plt.show()
