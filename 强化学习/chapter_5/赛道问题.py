import time
import numpy as np
import random
from collections import defaultdict
import pygame
import pickle
import os

def create_track_right():
    H, W = 31, 17  # 定义赛道尺寸：高度31，宽度17
    g = [["#"] * W for _ in range(H)]  # 初始化赛道，全部填充为墙（'#'）

    # 设置起点
    for j in range(3, 9):
        g[0][j] = "S"  # 在顶部行（y=0，x=3到8）设置起点

    # 设置终点
    for i in range(H - 1, H - 7, -1):
        g[i][-1] = "F"  # 在最后一列（x=16，y=24到30）设置终点线

    # 设置通道
    # 完整的竖线赛道
    for j in range(3, 9):
        for i in range(1, H):
            g[i][j] = "."  # 在第3到8列，从第1行到最后一行创建垂直赛道
    # 补全水平线
    for i in range(H - 1, H - 7, -1):
        for j in range(W - 2, 8, -1):
            g[i][j] = "."  # 在第24到30行，第9到15列创建水平赛道
    # 有一个单独的点
    g[H - 7][W - 8] = "."  # 在位置(24, 9)添加单个赛道点
    # 剩下的竖线
    for j in range(0, 3):
        if j == 1:
            for i in range(H - 4, H - 23, -1):
                g[i][j] = "."  # 在第1列，第8到27行创建垂直赛道
        elif j == 2:
            for i in range(H - 2, 2, -1):
                g[i][j] = "."  # 在第2列，第3到29行创建垂直赛道
        elif j == 0:
            for i in range(H - 5, H - 15, -1):
                g[i][j] = "."  # 在第0列，第16到26行创建垂直赛道
    return np.array(g)  # 将赛道转换为numpy数组并返回


# 赛车类
class RaceTrack:
    def __init__(self, track, noise_p=0.1, seed=42):
        self.rng = random.Random(seed)  # 使用指定种子初始化随机数生成器
        self.track = track  # 存储赛道布局
        self.H, self.W = track.shape  # 获取赛道尺寸
        self.noise_p = noise_p  # 动作噪声概率（加速度置零的概率）
        self.starts = [(x, y) for x in range(self.H) for y in range(self.W) if self.track[x][y] == "S"]  # 起点位置列表
        self.finishes = [(x, y) for x in range(self.H) for y in range(self.W) if self.track[x][y] == "F"]  # 终点位置列表
        # action为对速度赋予的加速度，动作集
        self.actions = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]  # 定义动作集为加速度对

    def random_start(self):
        # 随机选择其中一组元素
        x, y = self.rng.choice(self.starts)  # 随机选择一个起点位置
        # 返回当前起始位置与当前速度
        return (x, y, 0, 0)  # 返回起点位置和初始速度(0, 0)

    def _on_bounds(self, x, y):
        return 0 <= x < self.H and 0 <= y < self.W  # 检查位置是否在赛道边界内

    def _is_wall(self, x, y):
        return (not self._on_bounds(x, y)) or (self.track[x][y] == "#")  # 检查位置是否为墙或越界

    def _is_finish(self, x, y):
        # 假如没有在边界里面，那么就没有必要往后判断了
        return (self._on_bounds(x, y)) and self.track[x][y] == "F"  # 检查位置是否为终点

    def _on_start(self, x, y):
        return (self._on_bounds(x, y)) and self.track[x][y] == "S"  # 检查位置是否为起点

    def _rasterize_path(self, x, y, vx, vy):
        """
        vx,vy表示移动的速度，间接有了移动距离（x+vx, y+vy）为对应的终点，
        这里函数是对移动进行分解
        """
        path = []  # 初始化路径列表
        # 假如速度为0，则车仍在原点
        if vx == 0 and vy == 0:
            return [(x, y)]  # 如果速度为零，返回当前位置
        n = max(abs(vx), abs(vy))  # 根据最大速度分量确定步数
        # 将赛车按照单位距离进行移动，逐格拆解其是否会撞墙、越界、是否到达终点
        dx = vx / n  # x方向的增量步长
        dy = vy / n  # y方向的增量步长
        fx, fy = (float(x), float(y))  # 使用浮点数表示起始位置以精确计算
        for _ in range(n):
            fx += dx
            fy += dy
            path.append((int(round(fx)), int(round(fy))))  # 四舍五入到最近的网格位置
        return path  # 返回路径上的位置列表

    def _clip_velocity(self, vx, vy):
        """限制速度"""
        return max(0, min(vx, 4)), max(0, min(vy, 4))  # 将速度分量限制在[0, 4]

    def step(self, state, action, noisy=True):
        """
        :param state: 当前的状态（位置与速度）
        :param action: 预计施加的加速度
        :param noisy: 是否将其加速度置零
        :return: 返回下一时刻状态，奖励，是否结束该幕
        """
        x, y, vx, vy = state  # 解包当前状态（位置和速度）
        ax, ay = action  # 解包动作（加速度）
        # 两个方向上的加速度都有机会被重置为0
        if noisy and self.rng.random() <= self.noise_p:
            ax, ay = 0, 0  # 以noise_p概率将加速度置零
        # 将速度限制在5以内，防止赛车一直加速
        vx2, vy2 = self._clip_velocity(vx + ax, vy + ay)  # 更新并限制速度
        if (not self._on_start(x, y)) and vx2 == 0 and vy2 == 0:
            vx2 = 1  # 如果不在起点且速度为零，将x方向速度设为1
        path = self._rasterize_path(x, y, vx2, vy2)  # 根据新速度计算路径
        # 每移动一次，都给一个收益-1
        reward = -1  # 每步移动的奖励为-1
        for xx, yy in path:
            if self._is_finish(xx, yy):
                # print(xx, yy, vx2, vy2)
                return (xx, yy, vx2, vy2), reward, True  # 到达终点，返回新状态、奖励和完成标志
            if self._is_wall(xx, yy):
                """撞墙返回到起点重新开始"""
                start_x, start_y = self.rng.choice(self.starts)  # 撞墙后随机选择一个起点
                return (start_x, start_y, 0, 0), reward, False  # 返回新状态、奖励和未完成标志
        # 给出当前的终点
        nx, ny = path[-1]  # 获取路径的最后位置
        return (nx, ny, vx2, vy2), reward, False  # 返回新状态、奖励和未完成标志


# epsilon-策略
class EpsilonGreedyPolicy:
    def __init__(self, Q, actions, epsilon=0.1, seed=0):
        # Q：字典，存储状态-动作对
        self.Q = Q  # Q表，存储状态-动作值
        self.actions = actions  # 可用的动作列表
        self.rng = random.Random(seed)  # 使用指定种子初始化随机数生成器
        self.epsilon = epsilon  # epsilon值，控制探索与利用的平衡

    # epsilon策略
    def action(self, s):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)  # 以epsilon概率随机选择动作
        return self.greedy(s)  # 否则选择贪心动作

    # 一般策略
    def greedy(self, s):
        best_a = None
        best_q = float("-inf")
        for a in self.actions:
            q = self.Q.get((s, a), float("-inf"))  # 获取状态-动作对的Q值
            if q > best_q:
                best_q = q
                best_a = a  # 更新最佳动作
        if best_a is None:
            return self.rng.choice(self.actions)  # 如果没有最佳动作，随机选择
        return best_a  # 返回最佳动作


# 首次访问的蒙特卡洛预测（同轨）
def first_visit_mc_prediction(env, n_episodes=20000, epsilon=0.1, gamma=1.0, max_steps=20000, log_every=1000):
    Q = defaultdict(float)  # 初始化Q表，存储状态-动作值
    N = defaultdict(int)  # 初始化计数器，记录状态-动作对访问次数
    policy = EpsilonGreedyPolicy(Q, env.actions, epsilon)  # 创建epsilon-贪心策略

    # 每一幕采样
    def rollout(noisy=True):
        s = env.random_start()  # 随机选择起点
        traj = []  # 初始化轨迹列表
        for _ in range(max_steps):
            a = policy.action(s)  # 根据策略选择动作
            s2, r, done = env.step(s, a, noisy)  # 执行一步
            traj.append((s, a, r))  # 记录状态、动作、奖励
            s = s2  # 更新状态
            if done:
                traj.append((s, (0, 0), r))  # 如果结束，记录最终状态
                return traj
        return traj  # 返回轨迹

    # n_episodes幕进行采样
    for ep in range(1, n_episodes + 1):
        traj = rollout(noisy=True)  # 生成一幕轨迹
        G = 0.0  # 初始化累积回报
        visited = set()  # 记录访问过的状态-动作对
        for t in reversed(range(len(traj))):
            s, a, r = traj[t]
            # 价值累积
            G = gamma * G + r  # 更新累积回报
            if (s, a) not in visited:
                N[s, a] += 1  # 增加访问计数
                # 使用简单加权
                Q[s, a] += (G - Q[s, a]) / N[s, a]  # 更新Q值（增量平均）
                visited.add((s, a))  # 标记为已访问
        if ep % log_every == 0:
            print(f"Episode: {ep}, traj length: {len(traj)}")  # 每隔log_every幕打印信息
    return Q, N  # 返回Q表和访问计数


# 离轨策略，重要度采样
# 辅助函数
def greedy_action(Q, s, actions):
    rng = random.Random(0)  # 初始化随机数生成器
    best_a = None
    best_q = float("-inf")
    for a in actions:
        q = Q.get((s, a), float("-inf"))  # 获取状态-动作对的Q值
        if q > best_q:
            best_q = q
            best_a = a  # 更新最佳动作
    if best_a is None:
        return rng.choice(actions)  # 如果没有最佳动作，随机选择
    return best_a  # 返回最佳动作

def eps_soft_probs(Q, s, actions, epsilon=0.3):
    nA = len(actions)  # 动作数量
    # 填充一个概率向量，概率向量中，所有动作的概率都为 epsilon / nA
    p = np.full(nA, epsilon / nA, dtype=float)  # 初始化概率向量
    # 取最大值的索引
    gi = actions.index(greedy_action(Q, s, actions))  # 找到贪心动作的索引
    p[gi] += 1 - epsilon  # 为贪心动作增加概率
    return p  # 返回概率分布

# 随机采样一簇序列
def sample_from_prods(actions, probs, rng):
    r = rng.random()  # 生成随机数
    acc = 0.0
    for a, p in zip(actions, probs):
        acc += p
        if r <= acc:
            return a  # 根据概率选择动作
    return actions[-1]  # 返回最后一个动作（确保选择一个动作）

def off_policy_mc_prediction(env, n_episodes=20000, epsilon=0.3, gamma=1.0,
                            max_steps=20000, log_every=1000, seed=0):
    rng = random.Random(seed)  # 初始化随机数生成器
    actions = env.actions  # 获取动作集
    Q = defaultdict(float)  # 初始化Q表
    C = defaultdict(float)  # 初始化累积权重表

    for ep in range(1, n_episodes + 1):
        s = env.random_start()  # 随机选择起点
        episode = []  # 保存采样生成的轨迹：s,a,r,b
        # 用策略b生成一幕
        for _ in range(max_steps):
            # 由当前的Q来获取当前状态-动作对对应的概率向量b(a|s)
            b_probs = eps_soft_probs(Q, s, actions, epsilon)  # 计算行为策略的概率
            # 随机采样一个动作
            a = sample_from_prods(actions, b_probs, rng)  # 根据概率选择动作（离轨）随机选择一个策略
            s2, r, done = env.step(s, a, noisy=True)  # 执行一步
            b_prob = b_probs[actions.index(a)]  # 获取行为策略概率
            episode.append((s, a, r, b_prob))  # 记录状态、动作、奖励和概率
            s = s2  # 更新状态
            if done:
                break  # 如果到达终点，结束循环
        G = 0.0  # 初始化累积回报
        W = 1.0  # 初始化重要性权重
        # 对C的值做一些修正，防止上溢
        for t in reversed(range(len(episode))):
            s, a, r, b_prob = episode[t]  # 解包轨迹
            G = gamma * G + r  # 更新累积回报
            # 解出当前最优的pi(a|s)
            pi_probs = eps_soft_probs(Q, s, actions, epsilon=0.05)  # 计算目标策略概率
            # 根据a在actions中的索引，获取pi(a|s)
            pi_prob = pi_probs[actions.index(a)]  # 获取目标策略概率
            if pi_prob == 0:
                break  # 如果目标策略概率为0，停止更新
            # C 可能会上溢，也可为0
            r_t = pi_prob / b_prob  # 计算重要性权重比率
            # 限制pi/b
            C[s, a] = C[s, a] + W  # 更新累积权重
            if C[s, a] > 0:
                # 策略调整
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])  # 更新Q值
            W *= r_t  # 更新权重
            if W <= 1e-12:
                # 权重太小，继续训练已没有意义，跳出当前幕
                break
        if ep % log_every == 0:
            print(f"Episode: {ep}, W value: {W}")  # 每隔log_every幕打印信息
    # 用C来表示某些点是否被访问过，假如被访问过，C对应的值一定不为0
    return Q, C  # 返回Q表和累积权重表


# 渲染结果，最优路径（静态）
def render_path(track, path):
    vis = track.copy()  # 复制赛道用于可视化
    for (x, y, *_) in path:
        if 0 <= x < track.shape[0] and 0 <= y < track.shape[1]:
            if vis[x][y] == "." or vis[x][y] == "S" or vis[x][y] == "F":
                vis[x][y] = "o"  # 在路径点标记'o'
    print(vis)  # 打印可视化赛道
    print(path)  # 打印路径


# pygame绘制轨迹
def pygame_animate(track, path, cell_size=20, fps=5, show_trace=True):
    pygame.init()
    H, W = track.shape
    screen = pygame.display.set_mode((W * cell_size, H * cell_size))
    pygame.display.set_caption("赛车动画")
    clock = pygame.time.Clock()
    # 颜色映射
    colors = {
        "#": (0, 0, 0),       # 墙 黑
        ".": (255, 255, 255), # 道 白
        "S": (0, 255, 0),     # 起点 绿
        "F": (255, 0, 0),     # 终点 红
        "trace": (100, 149, 237), # 轨迹 蓝色 / CornflowerBlue
        "car": (0, 0, 255)    # 赛车 深蓝
    }
    # 轨迹集合
    visited = set()
    def draw(state=None):
        # 绘制赛道
        for i in range(H):
            for j in range(W):
                val = track[i, j]
                if val == "#":
                    color = colors["#"]
                elif val == ".":
                    color = colors["."]
                elif val == "S":
                    color = colors["S"]
                elif val == "F":
                    color = colors["F"]
                else:
                    color = (128, 128, 128)
                pygame.draw.rect(screen, color,
                                 (j * cell_size, i * cell_size, cell_size, cell_size))
        # 绘制轨迹
        if show_trace:
            for (x, y) in visited:
                pygame.draw.rect(screen, colors["trace"],
                                 (y * cell_size, x * cell_size, cell_size, cell_size))
        # 绘制赛车
        if state:
            x, y, *_ = state
            pygame.draw.rect(screen, colors["car"],
                             (y * cell_size, x * cell_size, cell_size, cell_size))
        pygame.display.flip()
    running = True
    step = 0
    while running and step < len(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # 添加到轨迹集合
        x, y, *_ = path[step]
        visited.add((x, y))
        draw(path[step])
        step += 1
        # 控制刷新频率 (fps 帧/秒)
        clock.tick(fps)
    # 保持窗口直到关闭
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)
    pygame.quit()


# 渲染结果，最优路径
def run_greedy_rollout(env, Q, max_steps=5000, noisy=False):
    s = env.random_start()  # 随机选择起点
    path = [s]  # 初始化路径
    for _ in range(max_steps):
        a = greedy_action(Q, s, env.actions)  # 选择贪心动作
        s2, r, done = env.step(s, a, noisy)  # 执行一步
        path.append(s2)  # 记录新状态
        s = s2  # 更新状态
        if done:
            return path, True  # 如果到达终点，返回路径和完成标志
    return path, False  # 返回路径和未完成标志


# 保存数据到本地
def save_data(track, path_off_policy, path, save_dir="data"):
    """
    将赛道和路径保存到本地
    :param track: 赛道布局（numpy数组）
    :param path_off_policy: 离轨策略路径
    :param path: 同轨策略路径
    :param save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建保存目录
    with open(os.path.join(save_dir, "track.pkl"), "wb") as f:
        pickle.dump(track, f)  # 保存赛道
    with open(os.path.join(save_dir, "path_off_policy.pkl"), "wb") as f:
        pickle.dump(path_off_policy, f)  # 保存离轨路径
    with open(os.path.join(save_dir, "path.pkl"), "wb") as f:
        pickle.dump(path, f)  # 保存同轨路径
    print(f"数据已保存到 {save_dir} 目录")


# 加载本地数据
def load_data(save_dir="data"):
    """
    从本地加载赛道和路径
    :param save_dir: 保存目录
    :return: track, path_off_policy, path
    """
    try:
        with open(os.path.join(save_dir, "track.pkl"), "rb") as f:
            track = pickle.load(f)
        with open(os.path.join(save_dir, "path_off_policy.pkl"), "rb") as f:
            path_off_policy = pickle.load(f)
        with open(os.path.join(save_dir, "path.pkl"), "rb") as f:
            path = pickle.load(f)
        print(f"从 {save_dir} 目录加载数据成功")
        return track, path_off_policy, path
    except FileNotFoundError:
        print(f"错误：无法找到 {save_dir} 目录下的数据文件，请先运行训练并保存数据")
        return None, None, None

# 主程序
def main():
    # 检查是否已有保存的数据
    save_dir = "data"
    if os.path.exists(os.path.join(save_dir, "track.pkl")):
        print("检测到已有数据，加载保存的数据...")
        track, path_off_policy, path = load_data(save_dir)
        if track is None:
            print("加载失败，将重新训练...")
            run_training = True
        else:
            run_training = False
    else:
        run_training = True

    if run_training:
        track = create_track_right()  # 创建赛道
        env = RaceTrack(track, noise_p=0.1, seed=1)  # 初始化赛车环境
        Q_off_policy, C = off_policy_mc_prediction(env, n_episodes=20000, epsilon=0.3, gamma=1.0, max_steps=10000, log_every=1000)  # 离轨蒙特卡洛预测
        path_off_policy, boolean = run_greedy_rollout(env, Q_off_policy)  # 运行贪心策略路径

        env = RaceTrack(track, noise_p=0.1, seed=1)  # 重新初始化环境
        Q, N = first_visit_mc_prediction(env, n_episodes=20000, epsilon=0.1, gamma=1.0, max_steps=5000, log_every=1000)  # 同轨蒙特卡洛预测
        path, boolean = run_greedy_rollout(env, Q)  # 运行贪心策略路径

        # 保存数据
        save_data(track, path_off_policy, path, save_dir)

    print("\noff_policy_mc Q path:")  # 打印离轨策略路径
    render_path(track, path_off_policy)  # 静态显示路径
    print("开始离轨策略动画...")
    pygame_animate(track, path_off_policy)  # 动态显示路径

    print("\nfirst_visit_mc Q path:")  # 打印同轨策略路径
    render_path(track, path)  # 静态显示路径
    print("开始同轨策略动画...")
    pygame_animate(track, path)  # 动态显示路径


if __name__ == "__main__":
    main()