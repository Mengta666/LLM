import numpy as np
import matplotlib.pyplot as plt

# 定义一个k臂赌博机类，用于模拟多臂赌博机问题
class k_arms_bandit:
    def __init__(self, k, is_non_stationary=False):
        # 初始化赌博机，设置臂的数量
        self.k = k  # 臂的数量
        # 为每个臂初始化真实的奖励值，从正态分布（均值=0，标准差=1）中随机抽取
        self.q_true = np.random.normal(0, 1, k)
        # 标志位，用于指示赌博机是否为非平稳（即真实奖励值是否随时间变化）
        self.is_non_stationary = is_non_stationary

    def run_bandit(self, action):
        # 模拟拉动选定的臂（action），返回一个奖励
        # 奖励从以该臂真实奖励值为均值的正态分布中抽取（标准差为1）
        reward = np.random.normal(self.q_true[action], 1)
        # 如果是非平稳赌博机，为所有臂的真实奖励值添加小的随机噪声
        if self.is_non_stationary:
            self.q_true += np.random.normal(0, 0.01, self.k)
        return reward

# 使用ε-贪婪策略或置信度上界来选择动作的函数
def get_greedy_action(Q, k, epsilon, is_UCB=False, t=None, N=None, C=2):
    if is_UCB:
        # 使用UCB策略选择动作
        result = []
        for i in range(k):
            if N[i] == 0:
                result.append(Q[i])
            else:
                result.append(Q[i] + C * np.sqrt(np.log(t) / N[i]))
        return np.argmax(np.array(result))
    # 以ε的概率随机选择一个动作（探索）
    if np.random.rand() < epsilon:
        return np.random.randint(0, k)
    # 否则选择估计奖励值最高的动作（利用）
    else:
        return np.argmax(Q)

# 运行一次赌博机实验，包含指定步数的试验
def run_bandit_once(epsilon: float, k=10, step=1000, is_non_stationary=False, alpha=0.1, is_optimistic=False, is_UCB=False, C=2):
    # 创建一个k臂赌博机实例
    bandit = k_arms_bandit(k, is_non_stationary)
    # 初始化每个臂的估计奖励值，初始为0
    if is_optimistic:
        Q = np.ones(k) * 5
    else:
        Q = np.zeros(k)
    # 存储每一步的奖励
    rewords = np.zeros(step)
    # 如果是平稳问题，初始化每个臂的选择次数
    if not is_non_stationary:
        N = np.zeros(k)

    # 运行指定步数的试验
    for t in range(step):
        # 根据相应策略选择动作
        if is_UCB:
            action = get_greedy_action(Q, k, epsilon, is_UCB, t, N, C)
        else:
            action = get_greedy_action(Q, k, epsilon)
        # 拉动选定臂，获取奖励
        reward = bandit.run_bandit(action)
        # 更新估计奖励值
        if not is_non_stationary:
            # 平稳问题使用样本平均法更新估计奖励
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
        else:
            # 非平稳问题使用固定步长α更新估计奖励
            Q[action] += (reward - Q[action]) * alpha
        # 记录当前步的奖励
        rewords[t] = reward
    return rewords

# 进行2000次赌博机的运行，所谓平均收益，指的是2000次运行完成以后，每一步机器的收益的平均值
def run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0.1, is_non_stationary=False, alpha=0.1, is_optimistic=True, is_UCB=False, C=2):
    # 初始化存储每次运行的奖励数组
    arevage_rewords = np.zeros([runs, step])
    # 运行指定次数的实验
    for i in range(runs):
        # 运行一次赌博机实验，获取奖励序列
        rewords = run_bandit_once(epsilon, k, step, is_non_stationary, alpha, is_optimistic, is_UCB, C)
        arevage_rewords[i] = rewords
    # 计算每一步的平均奖励（跨所有运行）
    return np.mean(arevage_rewords, axis=0)

if __name__ == '__main__':
    # -------- 主程序 --------
    steps = 1000  # 每次实验的步数
    runs = 2000   # 实验运行次数

    # 平稳问题，使用样本平均法，测试不同的ε值
    avg_rewards_stationary_01 = run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0.1,
                                                          is_non_stationary=False)
    avg_rewards_stationary_001 = run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0.01,
                                                           is_non_stationary=False)
    avg_rewards_stationary_0 = run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0,
                                                         is_non_stationary=False)
    # 非平稳问题，常数步长 α=0.1
    avg_rewards_nonstationary = run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0.1, alpha=0.1, is_non_stationary=True)

    # 乐观初始值, ε=0.1, 默认初始为5
    avg_rewards_stationary_01_opt = run_bandit_multiple_times(k=10, runs=2000, step=1000, epsilon=0.1,
                                                          is_non_stationary=False, is_optimistic=True)
    # 置信度上界的动作选择
    avg_rewards_stationary_UCB = run_bandit_multiple_times(k=10, runs=2000, step=1000,
                                                              is_non_stationary=False, is_UCB=True, C=2)

    # -------- 绘图 --------
    plt.figure(figsize=(8, 5))
    plt.plot(avg_rewards_stationary_01, label="Stationary, epsilon=0.1")
    plt.plot(avg_rewards_stationary_0, label="Stationary, epsilon=0")
    plt.plot(avg_rewards_stationary_001, label="Stationary, epsilon=0.01")
    plt.plot(avg_rewards_nonstationary, label="Non-stationary, epsilon=0.1, alpha=0.1")
    plt.plot(avg_rewards_stationary_01_opt, label="Stationary, epsilon=0.1, is_optimistic=True")
    plt.plot(avg_rewards_stationary_UCB, label="Stationary, epsilon=0, is_UCB=True")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("10-armed bandit testbed")
    plt.show()

