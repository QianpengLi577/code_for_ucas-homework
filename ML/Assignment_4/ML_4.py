# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 16:04
# @Author  : Qianpeng Li
# @FileName: ML_4.py
# @Contact : liqianpeng2021@ia.ac.cn

import gym
import numpy as np


def e_policy(state, e, Q, step):
    # choose a action at state using e-policy
    action_space = Q.shape[1]
    if np.random.rand(1) < e/(1+step/10):
        return np.random.randint(action_space)
    else:
        return np.argmax(Q[state, :])


def train(env, alpha, gama, e, iters, steps, scaling, method):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    reward_list = []
    for i in range(iters):
        state = env.reset()
        action = e_policy(state, e, Q, 0)
        done = False
        step = 0
        while not done:
            state_next, reward, done, info = env.step(action)
            reward *= scaling
            action_next = e_policy(state_next, e, Q, step)
            if method == 'sarsa':
                Q[state, action] = Q[state, action]+alpha * \
                    (reward+int(not done)*gama *
                     Q[state_next, action_next]-Q[state, action])
            elif method == 'Q_learning':
                Q[state, action] = Q[state, action]+alpha * \
                    (reward+int(not done)*gama *
                     Q[state_next, :].max()-Q[state, action])
            else:
                print('')
            state = state_next
            action = action_next
            step += 1
            if(step > steps):
                done = True
        # reward_list.append(test(env, False, Q, ' ', 50))
    return Q, reward_list


def test(env, log, Q, method, test_steps):
    reward_list = []
    for _ in range(test_steps):
        state = env.reset()
        action = np.argmax(Q[state, :])
        done = False
        reward_sum = 0
        # env.render()
        while not done:
            state, reward, done, info = env.step(action)
            reward_sum += reward
            # env.render()
            action = np.argmax(Q[state, :])
        reward_list.append(reward_sum)
    if log:
        print(method + ' finsh')
        print('average reward: ', np.array(reward_list).sum()/len(reward_list))
    else:
        return np.array(reward_list).sum()/len(reward_list)
    # print(info)


env = gym.make('FrozenLake-v1')
print('observation_space:', env.observation_space.n)
print('action_space:', env.action_space.n)
alpha, e, gama, iters, steps, scaling, test_steps = 0.01, 0.2, 0.99, 50000, 200, 5, 200
Q_sarsa, r_sarsa = train(env, alpha, gama, e, iters, steps, scaling, 'sarsa')
test(env, True, Q_sarsa, 'sarsa', test_steps)

Q_Q_learning, r_q = train(env, alpha, gama, e, iters,
                          steps, scaling, 'Q_learning')
test(env, True, Q_Q_learning, 'Q_learning', test_steps)
