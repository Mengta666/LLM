# 书中的赌徒问题，p82
# 使用价值迭代

import numpy as np
import matplotlib.pyplot as plt


# 全局变量
GAMMA = 1.0
ph = 0.4          # 抛硬币正面朝上的概率

def q_value_state(V, s, a):
    """
    V: 状态价值函数
    s: 当前状态，也就是当前所拥有的赌资
    a: 下赌注的金额
    return: 返回的是当前状态的最新价值函数
    """
    r = 0.0
    if (s + a) == 100:
        r = 1.0
    return ph * (r + GAMMA * V[s+a]) + (1-ph) * (GAMMA * V[s-a])

THETA = 1e-4
def value_iteration():
    # 只用1-101的索引，0号索引不使用
    V = np.zeros(101, dtype=float)
    policy = np.zeros(101, dtype=int)
    print("Start value iteration...")
    while True:
        delta = 0.0
        V_new = V.copy()
        for s in range(1, 100):
            q_best = 1e-12
            for a in range(1, min(s, 100-s)+1):
                q = q_value_state(V, s, a)
                if q > q_best:
                    q_best = q
            V_new[s] = q_best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < THETA:
            break
    for s in range(1, 100):
            q_best = 1e-12
            a_best = 100
            for a in range(1, min(s, 100-s)+1):
                q = q_value_state(V, s, a)
                if (q > q_best + 1e-12) or (abs(q - q_best) < 1e-12) and (abs(a) < abs(a_best)):
                    a_best, q_best = a, q
            policy[s] = a_best
    print("Finish value iteration...")
    V[100] = 1.0
    return V, policy

def plot_V_policy(V, policy):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 左边价值函数
    axes[0].plot(range(101), V)
    axes[0].set_title("Value Function")
    axes[0].set_xlabel("Capital")
    axes[0].set_ylabel("Value")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1.1)  # V(s) 是到达100的概率，最大为1
    # 右边最优策略
    axes[1].plot(range(101), policy, drawstyle="steps-mid")
    axes[1].set_title("Final Policy (stake)")
    axes[1].set_xlabel("Capital")
    axes[1].set_ylabel("Stake")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 55)  # 最大下注额不会超过50
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    V, policy = value_iteration()
    plot_V_policy(V, policy)
    ph = 0.25
    V, policy = value_iteration()
    plot_V_policy(V, policy)
    ph = 0.55
    V, policy = value_iteration()
    plot_V_policy(V, policy)




