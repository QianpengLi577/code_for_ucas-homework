# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 18:27
# @Author  : Qianpeng Li
# @FileName: analysis.py
# @Contact : liqianpeng2021@ia.ac.cn
# @Github  : https://github.com/QianpengLi577

from ML_4 import e_policy, train, test
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')
alpha_list = [0.02, 0.04, 0.08, 0.1, 0.2]
gamma_list = [0.59, 0.69, 0.79, 0.89, 0.99]
scaling_list = [1, 2, 3, 4, 5]
# Initialize gym and print some related information
alpha, e, gamma, iters, steps, scaling, test_steps = 0.1, 0.4, 0.99, 20000, 200, 5, 1000
# differert aplha using sarsa
R1 = []
for i in range(len(alpha_list)):
    Q_sarsa, r_sarsa = train(env, alpha_list[i], gamma, e, iters,
                            steps, scaling, 'sarsa', True)
    R1.append(r_sarsa)


# differert aplha using Q_L
R2 = []
for i in range(len(alpha_list)):
    Q_Q_learning, r_q = train(env, alpha_list[i], gamma, e, iters,
                            steps, scaling, 'Q_learning', True)
    R2.append(r_q)


# differert gamma using sarsa
R3 = []
for i in range(len(gamma_list)):
    Q_sarsa, r_sarsa = train(env, alpha, gamma_list[i], e, iters,
                            steps, scaling, 'sarsa', True)
    R3.append(r_sarsa)



# differert gamma using Q_L
R4 = []
for i in range(len(gamma_list)):
    Q_Q_learning, r_q = train(env, alpha, gamma_list[i], e, iters,
                            steps, scaling, 'Q_learning', True)
    R4.append(r_q)



# differert aplha using sarsa
R5 = []
for i in range(len(scaling_list)):
    Q_sarsa, r_sarsa = train(env, alpha, gamma, e, iters,
                            steps, scaling_list[i], 'sarsa', True)
    R5.append(r_sarsa)



# differert aplha using Q_L
R6 = []
for i in range(len(scaling_list)):
    Q_Q_learning, r_q = train(env, alpha, gamma, e, iters,
                            steps, scaling_list[i], 'Q_learning', True)
    R6.append(r_q)



np.savetxt('R1.csv', R1, delimiter=',')
np.savetxt('R2.csv', R2, delimiter=',')
np.savetxt('R3.csv', R3, delimiter=',')
np.savetxt('R4.csv', R4, delimiter=',')
np.savetxt('R5.csv', R5, delimiter=',')
np.savetxt('R6.csv', R6, delimiter=',')

R1 = np.loadtxt('R1.csv',dtype=np.float32, delimiter=',')
R2 = np.loadtxt('R2.csv',dtype=np.float32, delimiter=',')
R3 = np.loadtxt('R3.csv',dtype=np.float32, delimiter=',')
R4 = np.loadtxt('R4.csv',dtype=np.float32, delimiter=',')
R5 = np.loadtxt('R5.csv',dtype=np.float32, delimiter=',')
R6 = np.loadtxt('R6.csv',dtype=np.float32, delimiter=',')

x = np.linspace(0, R1.shape[1], R1.shape[1])

plt.figure(figsize=(10, 6))
plt.title('sarsa with different alpha')
for i in range(R1.shape[0]):
    plt.plot(x,R1[i,:],label='alpha-'+str(alpha_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('sarsa with different alpha.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.title('Q-learning with different alpha')
for i in range(R2.shape[0]):
    plt.plot(x,R2[i,:],label='alpha-'+str(alpha_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('Q-learning with different alpha.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.title('sarsa with different gamma')
for i in range(R3.shape[0]):
    plt.plot(x,R3[i,:],label='gamma-'+str(gamma_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('sarsa with different gamma.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.title('Q-learning with different gamma')
for i in range(R4.shape[0]):
    plt.plot(x,R4[i,:],label='gamma-'+str(gamma_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('Q-learning with different gamma.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.title('sarsa with different reward')
for i in range(R5.shape[0]):
    plt.plot(x,R5[i,:],label='reward-'+str(scaling_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('sarsa with different reward.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.title('Q-learning with different reward')
for i in range(R6.shape[0]):
    plt.plot(x,R6[i,:],label='reward-'+str(scaling_list[i]),ls=":")
plt.legend()
plt.xlabel('train step')
plt.ylabel('average reward')
plt.savefig('Q-learning with different reward.png')
plt.show()

print('finsh')
