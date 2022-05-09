# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 11:58
# @Author  : Qianpeng Li
# @FileName: cliff_wk.py
# @Contact : liqianpeng2021@ia.ac.cn
# @Github  : https://github.com/QianpengLi577

# this file is an example for cliff_walking using sarsa and Q-learning to test function is right
from ML_4 import e_policy, train, test
import gym
import numpy as np

env = gym.make('CliffWalking-v0')
print('observation_space:', env.observation_space.n)
print('action_space:', env.action_space.n)
alpha, e, gamma, iters, steps, scaling, test_steps = 0.1, 0.4, 0.99, 20000, 200, 5, 1
Q_sarsa, r_sarsa = train(env, alpha, gamma, e, iters,
                         steps, scaling, 'sarsa', False)
test(env, True, Q_sarsa, 'sarsa', test_steps, plot=True)

Q_Q_learning, r_q = train(env, alpha, gamma, e, iters,
                          steps, scaling, 'Q_learning', False)
test(env, True, Q_Q_learning, 'Q_learning', test_steps, plot=True)

# the result will be like this

# observation_space: 48
# action_space: 4
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# x  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# x  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# x  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# x  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  x  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  x  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  x  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  x  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  x  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  x  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  x  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  x  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  x  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  x  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  x
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  x
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  x
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  x

# sarsa finsh
# average reward:  -17.0
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# x  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# x  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  x  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  x  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  x  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  x  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  x  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  x  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  x  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  x  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  x  o  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  x  o
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  x
# o  C  C  C  C  C  C  C  C  C  C  T

# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  o  o  o  o  o  o  o  o  o  o  o
# o  C  C  C  C  C  C  C  C  C  C  x

# Q_learning finsh
# average reward:  -13.0
