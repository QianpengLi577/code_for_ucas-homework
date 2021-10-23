import numpy as np
import random
import time
import math
import copy


# 模拟退火算法解决八皇后问题
# 八数码初始化函数，返回一个初始状态和一个目标状态,这里0代表八数码中的空格
def init():
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    target_state = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.random.shuffle(init_state)
    np.random.shuffle(target_state)
    init_state = np.reshape(init_state, (3, 3))
    target_state = np.reshape(target_state, (3, 3))
    return init_state, target_state


# 计算当前状态和目标状态下的曼哈顿距离
def compute_manhattan_distance(init_state, target_state):
    total_distance = 0
    for i in range(1, 9):
        (init_row, init_column) = np.where(init_state == i)
        (target_row, target_column) = np.where(target_state == i)
        total_distance += abs(target_row - init_row) + abs(target_column - init_column)
    return int(total_distance)


# 计算接下来所有可以移动的方向，因为空白方块可能处于边界
def get_direction(state):
    current_state = state.copy()
    coord_cache = {}
    zero_row, zero_column = np.where(current_state == 0)
    if zero_row - 1 >= 0:  # 如果空白方块在最上边则不能往上移，下面同理
        up_row, up_column = zero_row - 1, zero_column
        coord_cache["up"] = [up_row, up_column]
    if zero_row + 1 <= 2:
        down_row, down_column = zero_row + 1, zero_column
        coord_cache["down"] = [down_row, down_column]
    if zero_column - 1 >= 0:
        left_row, left_column = zero_row, zero_column - 1
        coord_cache["left"] = [left_row, left_column]
    if zero_column + 1 <= 2:
        right_row, right_column = zero_row, zero_column + 1
        coord_cache["right"] = [right_row, right_column]
    return coord_cache


# 随机产生一个新的解
def random_adjust(state, target_state):
    # 随机选择可以移动的方向把坐标取出来
    current_state = state.copy()
    direction = get_direction(state)
    cache = list(direction.keys())
    index = random.randrange(0, len(cache))
    row, column = direction[cache[index]]
    row, column = int(row), int(column)  # fix bug,这里的row，column等是列表，直接作为索引使用会报错，转换为int类型
    # 开始随机调整
    zero_row, zero_column = np.where(current_state == 0)
    zero_row, zero_column = int(zero_row), int(zero_column)
    temp = current_state[row][column]
    current_state[zero_row][zero_column] = temp
    current_state[row][column] = 0
    distance = compute_manhattan_distance(current_state, target_state)
    return current_state, distance


# 模拟退火算法
def sa_algorithm(temperature, temperature_min, r, L):
    """
    :param temperature: 初始温度值
    :param r:用于控制降温的快慢
    :param temperature_min:温度下限，低于这个温度还没有找到最优解算法结束
    :param L:每个温度的迭代次数
    :return:bool
    """
    state, target_state = init()
    print("目标状态为：\n", target_state)
    while temperature > temperature_min:
        for j in range(L):  # 每个温度循环L次
            current_manhattan_distance = compute_manhattan_distance(state, target_state)
            print("当前状态为：\n", state)
            print("当前状态的曼哈顿距离为：", current_manhattan_distance)
            if current_manhattan_distance == 0:  # 如果曼哈顿距离为零，则代表就是目标状态了
                return True
            new_state, new_manhattan_distance = random_adjust(state, target_state)  # 随机调整得到一个新的解和这个解的曼哈顿距离
            print("随机产生的新解状态为：\n", new_state)
            print("随机产生新解的曼哈顿距离为：", new_manhattan_distance)
            delta_manhattan_distance = current_manhattan_distance - new_manhattan_distance
            if delta_manhattan_distance > 0:  # 如果新解更好 接受
                print("这是一个更好的解，直接接收")
                state = new_state
            else:
                if math.exp(delta_manhattan_distance / temperature) > random.random():  # 如果新解更差 以模拟退火的概率接受它
                    print("当前接收更差解的概率为：", math.exp(delta_manhattan_distance / temperature))
                    print("这是一个更差的解，但是被接收了")
                    state = new_state
        temperature = temperature * r  # 模拟退火，温度衰减
    return False


def SA_algorithm_test(temperature, temperature_min, r, L, num):
    tic = time.time()
    success_case = 0
    fail_case = 0
    for i in range(num):
        if sa_algorithm(temperature, temperature_min, r, L):
            success_case += 1
            # time.sleep(10000)
            print("第{0}个例子找到了最优解".format(i))
        else:
            fail_case += 1
            print("第{0}个例子失败".format(i))
    toc = time.time()
    print("{0}个例子中成功解决的例子为：{1}".format(num, success_case))
    print("{0}个例子成功解决的百分比为：{1}".format(num, success_case / num))
    print("{0}个例子中失败的例子为：{1}".format(num, fail_case))
    print("{0}个例子失败的百分比为：{1}".format(num, fail_case / num))
    print("{0}个例子运行算法所需的时间为：{1}秒".format(num, toc - tic))


SA_algorithm_test(temperature=5, temperature_min=0.001, r=0.8, L=150, num=100)