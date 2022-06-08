import numpy as np
import queue

op_list = np.array([[0, 2], [0, 1], [1, 1], [1, 0], [2, 0]])  # 船上人员可以存在的集合，第一维为传教士，第二维为野人


def inside(x, a, b):  # a<=x<=b
    if ((x >= a) & (x <= b)):
        return 1
    else:
        return 0


def get_nextstatus(status, op_list):  # 产生下一个状态待选集合
    next_status = []
    a = []
    C = status[0]
    Y = status[1]
    B = status[2]
    for i in range(op_list.shape[0]):
        if (B == 1):
            if (inside(C - op_list[i, 0], 0, 3) & inside(Y - op_list[i, 1], 0, 3) & inside(3 - C + op_list[i, 0], 0,
                                                                                           3) & inside(
                3 - Y + op_list[i, 1], 0, 3)):
                if (((((C - op_list[i, 0]) >= (Y - op_list[i, 1])) & (
                        (3 - C + op_list[i, 0]) >= (3 - Y + op_list[i, 1]))) | (
                             C - op_list[i, 0] == 0) | (3 - C + op_list[i, 0] == 0))):
                    next_status.append([C - op_list[i, 0], Y - op_list[i, 1], 0])
        else:
            if (inside(C + op_list[i, 0], 0, 3) & inside(Y + op_list[i, 1], 0, 3) & inside(3 - C - op_list[i, 0], 0,
                                                                                           3) & inside(
                3 - Y - op_list[i, 1], 0, 3)):
                if ((((C + op_list[i, 0]) >= (Y + op_list[i, 1])) & (
                        (3 - C - op_list[i, 0]) >= (3 - Y - op_list[i, 1]))) | (
                        3 - C - op_list[i, 0] == 0) | (C + op_list[i, 0] == 0)):
                    next_status.append([C + op_list[i, 0], Y + op_list[i, 1], 1])
    return next_status


class agent():  # agent定义：包括father以及当前状态
    def __init__(self, status):
        self.father = None
        self.status = status

    def getfather(self, father):  # father赋值
        self.father = father


father_status = [3, 3, 1]  # 初试化状态
open = queue.Queue()
close = queue.Queue()  # open和close表
closed_list = []  # 探索过节点的list，用于防止重复探索
show_list = []  # 探索路径的展示--需要通过father一步一步回调

father = agent(father_status)  # 生成father
open.put(father)  # open初试化
while (~open.empty()):
    now = open.get()
    close.put(now)  # 取open以及放入close
    if now.status == [0, 0, 0]:
        print('结束')
        show_list.append(now.status)
        while (now.father != None):
            now = now.father
            show_list.append(now.status)
        print('转移过程：')
        for i in range(len(show_list)):
            print(show_list[len(show_list) - 1 - i])
        break
    if (closed_list.count(now.status) == 0):
        next_status = get_nextstatus(now.status, op_list)
        print('当前状态:', now.status)
        print('子节点可能有的状态:', next_status)
        for x in next_status:
            if (closed_list.count(x) == 0):
                if (now.father != None):
                    if (x != now.father.status):
                        a = agent(x)
                        a.getfather(now)
                        open.put(a)
                else:
                    a = agent(x)
                    a.getfather(now)
                    open.put(a)  # 生成子节点依赖关系，初始状态注意需要特殊处理
        closed_list.append(now.status)  # 扩展已经探索过的状态
