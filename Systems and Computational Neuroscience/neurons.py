# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 20:03
# @Author  : Qianpeng Li
# @FileName: neurons.py
# @Contact : liqianpeng2021@ia.ac.cn
import matplotlib.pyplot as plt
import numpy as np


class LIF():
    def __init__(self, vth, tau, rst, t_rest, r_m):
        # LIF神经元模型
        # Reference  Gerstner W, Kistler WM (2002). " Spiking neuron models : single neurons, populations, plasticity "
        # 参数
        # vth 阈值电压
        # rst 重置电压
        # tau 时间常数
        # v   膜电压
        # t_rest 不应期时间
        # r_m 等效电阻

        self.vth = vth
        self.tau = tau
        self.rst = rst
        self.v = 0
        self.t_rest = t_rest
        self.r_m = r_m

    def work(self, I, T, dt):
        # 参数
        # I 输入的标准电流(时变)
        # T 仿真时长
        # dt 时间步
        v = []  # 电压的记录
        t_pause = 0  # 发放脉冲的时间
        flag = 0  # 可以接受电流
        for i in range(int(T / dt)):
            dv = dt * ((self.r_m * I[i] - self.v) / self.tau)  # 电压变化量
            if (flag == 0):
                if (self.v >= self.vth):
                    self.v = self.vth + 0.5  # 用于显示发放脉冲的动作
                    flag = 1  # 不应期
                    t_pause = i  # 发放脉冲的时间，用于计算什么时候能接受电流
                else:
                    self.v = self.v + dv
            elif (t_pause + self.t_rest / dt <= i):  # 不应期已过
                flag = 0
                self.v = dv + self.v
            else:  # 仍处不应期
                self.v = 0
            v.append(self.v)
        return v


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
plt.show()


class Izh():
    # Izh神经元模型
    # Reference：E. M. Izhikevich, "Simple model of spiking neurons,"
    def __init__(self, a, b, c, d):
        # 参数
        # a b c d 与发放脉冲的频率模式有关
        # v 电压
        # u 恢复变量
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = -65
        self.u = 0

    def work(self, I, T, dt):
        # 参数
        # I 输入电流
        # T 仿真总时间
        # dt 仿真步长
        v = []
        for i in range(int(T / dt)):
            dv = (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + I[i]) * dt
            du = dt * self.a * (self.b * self.v - self.u)
            self.u = self.u + du
            self.v = self.v + dv
            if (self.v >= 30):
                self.v = self.c
                self.u = self.u + self.d
            v.append(self.v)
        return v


T = 3000
I = 4 * np.ones(int(T / dt))
node2 = Izh(0.02, 0.2, -65, 8)
v = node2.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-RS')
plt.show()

I = 4 * np.ones(int(T / dt))
node3 = Izh(0.02, 0.2, -50, 2)
v = node3.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-CH')
plt.show()

I = 4 * np.ones(int(T / dt))
node4 = Izh(0.1, 0.2, -65, 8)
v = node4.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 8)], v[0:int(T / dt / 8)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('Izh-FS')
plt.show()


class HH():
    # HH 模型
    # Reference： HODGKIN AL, HUXLEY AF. " A quantitative description of membrane current and its application to conduction and excitation in nerve. "
    def __init__(self, g_na, g_k, g_l):
        # 参数
        # g_na 纳电导
        # g_k 钾电导
        # g_l 漏电导
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.v = -70  # 电压初始化

    def work(self, I, T, dt):
        # 参数
        # I 输入电流
        # T 仿真总时间
        # dt 仿真步长
        # m n h alpha_m/n/h beta_m/n/h C v_k/na/l 为模型必要的参数
        m = 0
        n = 0
        h = 0
        v_k = -12
        v_na = 115
        v_l = 10.6
        C = 1.0
        v = []
        for i in range(int(T / dt)):
            alpha_m = (2.5 - 0.1 * self.v) / (-1 + np.exp(2.5 - 0.1 * self.v))
            beta_m = 4 * np.exp(-self.v / 18)
            alpha_n = (0.1 - 0.01 * self.v) / (-1 + np.exp(1 - 0.1 * self.v))
            beta_n = 0.125 * np.exp(-self.v / 80)
            alpha_h = 0.07 * np.exp(-self.v / 20)
            beta_h = 1 / (1 + np.exp(3 - 0.1 * self.v))
            if (i == 1):
                m = alpha_m / (alpha_m + beta_m)
                n = alpha_n / (alpha_n + beta_n)
                h = alpha_h / (alpha_h + beta_h)
            dm = dt * (alpha_m * (1 - m) - beta_m * m)
            dn = dt * (alpha_n * (1 - n) - beta_n * n)
            dh = dt * (alpha_h * (1 - h) - beta_h * h)
            dv = dt * (I[i] - self.g_l * (self.v - v_l) - self.g_k * np.power(n, 4) * (
                        self.v - v_k) - self.g_na * np.power(m, 3) * h * (self.v - v_na)) / C

            self.v = self.v + dv
            m = m + dm
            n = n + dn
            h = h + dh
            v.append(self.v)
        return v


I = 2 * np.ones(int(T / dt))
node5 = HH(120, 36, 0.3)
v = node5.work(I, T, dt)
plt.figure()
plt.plot(np.arange(0, T, dt)[0:int(T / dt / 40)], v[0:int(T / dt / 40)])
plt.xlabel('time / ms')
plt.ylabel('voltage / v')
plt.title('HH')
plt.show()
