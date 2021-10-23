class ENV():
    def __init__(self, loc, dis):
        self.loc = loc  # 环境的位置
        self.dis = dis  # 环境是否有垃圾，0代表无垃圾，1代表有垃圾

    def getloc(self):
        return self.loc

    def getdis(self):
        return self.dis


class agent():
    def __init__(self, reword, cost_clean, cost_run, home):
        self.reword = reword  # 正确判断的奖赏
        self.cost_clean = cost_clean  # 清扫的电量消耗
        self.cost_run = cost_run  # 移动的电量消耗
        self.home = home  # 初始位置
        self.location = home  # 起始位置在家
        self.perf = 0.0

    def print_perf(self):
        print("性能：", self.perf)

    def sensor(self, env):
        if env:
            return 1
        else:
            return 0

    # env代表输入(0无垃圾，1有垃圾)，0代表没有垃圾，1代表有垃圾
    def action(self, enva, envb):
        if (self.location == enva.loc):
            if (self.sensor(enva.dis) == 1):
                self.perf = self.perf + self.cost_clean  # 当前有垃圾，需要耗电clean
            self.location = envb.loc
            self.perf = self.perf + self.cost_run + self.reword  # 转移到另一个env，耗电run，并得到奖赏
            if (self.sensor(envb.dis) == 1):
                self.perf = self.perf + self.cost_clean
            self.perf = self.perf + self.reword
        else:
            if (self.sensor(envb.dis) == 1):
                self.perf = self.perf + self.cost_clean
            self.location = enva.loc
            self.perf = self.perf + self.cost_run + self.reword
            if (self.sensor(enva.dis) == 1):
                self.perf = self.perf + self.cost_clean
            self.perf = self.perf + self.reword
        if (self.location != self.home):
            self.location = self.home
            self.perf = self.perf + self.cost_run  # 最终需要返回到home，在下一个仿真步判断需不需要清扫此时的home


env00 = ENV(0, 0)
env01 = ENV(0, 1)
env10 = ENV(1, 0)
env11 = ENV(1, 1)
a = agent(10, -2, -1, 0)
a.action(env00, env10)
a.print_perf()
a = agent(10, -2, -1, 0)
a.action(env00, env11)
a.print_perf()
a = agent(10, -2, -1, 0)
a.action(env01, env10)
a.print_perf()
a = agent(10, -2, -1, 0)
a.action(env01, env11)
a.print_perf()
a = agent(10, -2, -1, 1)
a.action(env00, env10)
a.print_perf()
a = agent(10, -2, -1, 1)
a.action(env00, env11)
a.print_perf()
a = agent(10, -2, -1, 1)
a.action(env01, env10)
a.print_perf()
a = agent(10, -2, -1, 1)
a.action(env01, env11)
a.print_perf()
