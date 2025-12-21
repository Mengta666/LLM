import random
from collections import defaultdict
import pygame


class WindyGridworld:
    """
    风力格子世界环境，用于强化学习实验，模拟一个有风影响的网格环境。
    智能体需从起点移动到目标点，风力会影响移动轨迹。
    """

    def __init__(self, n_row=7, n_col=10, start=(3, 0), end=(3, 7),
                 wind=(0, 0, 0, 1, 1, 1, 2, 2, 1, 0), random_windy=False):
        """
        初始化风力格子世界环境。

        参数:
            n_row (int): 网格行数，默认为7。
            n_col (int): 网格列数，默认为10。
            start (tuple): 起点坐标 (row, col)，默认为(3,0)。
            end (tuple): 目标坐标 (row, col)，默认为(3,7)。
            wind (tuple): 每列的风力值，向上吹（负方向），默认为(0,0,0,1,1,1,2,2,1,0)。
            random_windy (bool): 是否启用随机风力扰动，默认为False。
        """
        self.n_row = n_row
        self.n_col = n_col
        self.start = start
        self.end = end
        self.wind = list(wind)
        self.random_windy = random_windy
        self.reset_start = start

    def reset(self):
        """
        重置环境，将智能体位置设为起点。

        返回:
            tuple: 起点坐标 (row, col)。
        """
        self.reset_start = self.start
        return self.reset_start

    def inside(self, now_r, now_c):
        """
        确保智能体位置在网格边界内。

        参数:
            now_r (int): 当前行坐标。
            now_c (int): 当前列坐标。

        返回:
            tuple: 修正后的坐标 (row, col)，限制在网格内。
        """
        r = max(0, min(self.n_row - 1, now_r))
        c = max(0, min(self.n_col - 1, now_c))
        return r, c

    def step(self, a):
        """
        执行一步动作，更新智能体位置，考虑风力影响。

        参数:
            a (tuple): 动作，格式为 (dr, dc)，表示行和列的变化量。

        返回:
            tuple: (新状态, 奖励, 是否终止)。
                - 新状态: (row, col)，智能体的新位置。
                - 奖励: 每步奖励，固定为-1。
                - 是否终止: 布尔值，是否到达目标点。
        """
        (r, c) = self.reset_start
        dr, dc = a
        r2, c2 = self.inside(r + dr, c + dc)
        base = self.wind[c2]
        w = base
        if self.random_windy:
            w = base + random.choice([-1, 0, 1])  # 随机风力扰动
        r2, c2 = self.inside(r2 - w, c2)  # 风力向上吹（负方向）
        self.reset_start = (r2, c2)
        done = (self.reset_start == self.end)
        reward = -1
        return self.reset_start, reward, done

    def actions4(self):
        """
        返回四方向动作集（上、下、左、右）。

        返回:
            list: 动作列表，每个动作是 (dr, dc) 元组。
        """
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def actions8(self):
        """
        返回八方向动作集（上、下、左、右及四个对角方向）。

        返回:
            list: 动作列表，每个动作是 (dr, dc) 元组。
        """
        return [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, 1), (1, -1), (-1, 1)]

    def actions9(self):
        """
        返回九方向动作集（八方向加静止动作）。

        返回:
            list: 动作列表，每个动作是 (dr, dc) 元组。
        """
        return [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, 1), (1, -1), (-1, 1),
                (0, 0)]


def epsilon_greedy(Q, s, actions, eps):
    """
    ε-贪婪策略选择动作。

    参数:
        Q (defaultdict): 状态-动作值函数。
        s (tuple): 当前状态 (row, col)。
        actions (list): 可选动作列表。
        eps (float): 探索概率 ε。

    返回:
        tuple: 选择的动作 (dr, dc)。
    """
    if random.random() < eps:
        return random.choice(actions)
    best_a, best_q = None, float("-inf")
    for a in actions:
        q = Q[s, a]
        if q > best_q:
            best_q = q
            best_a = a
    if best_a is None:
        best_a = actions[-1]
    return best_a


def saras(env: WindyGridworld, actions: list[tuple], episodes=8000,
          alpha=0.5, gamma=1.0, epsilon=0.1, log_every=1000, seed=0):
    """
    SARSA 强化学习算法，训练状态-动作值函数 Q。

    参数:
        env (WindyGridworld): 风力格子世界环境。
        actions (list): 可选动作列表。
        episodes (int): 训练幕数，默认为8000。
        alpha (float): 学习率，默认为0.5。
        gamma (float): 折扣因子，默认为1.0。
        epsilon (float): ε-贪婪策略的探索概率，默认为0.1。
        log_every (int): 每隔多少幕打印平均步数，默认为1000。
        seed (int): 随机种子，默认为0。

    返回:
        defaultdict: 训练后的 Q 值表。
    """
    random.seed(seed)
    Q = defaultdict(float)
    steps_history = []
    for ep in range(1, episodes + 1):
        s = env.reset()
        a = epsilon_greedy(Q, s, actions, epsilon)
        steps = 0
        while True:
            s2, r, done = env.step(a)
            steps += 1
            if done:
                td_target = r
                Q[(s, a)] = Q[(s, a)] + alpha * (td_target - Q[(s, a)])
                steps_history.append(steps)
                break
            else:
                a2 = epsilon_greedy(Q, s2, actions, epsilon)
                td_target = r + gamma * Q[(s2, a2)]
                Q[(s, a)] += alpha * (td_target - Q[(s, a)])
                s, a = s2, a2
        if log_every and (ep % log_every == 0):
            avg = sum(steps_history) / log_every
            print(f"[{ep:>5d}] 最近 {log_every} 幕平均步数: {avg:.2f}")
            steps_history = []
    return Q


def greedy_rollout(env: WindyGridworld, Q, actions, max_steps=500):
    """
    使用贪婪策略从起点到目标点生成路径。

    参数:
        env (WindyGridworld): 风力格子世界环境。
        Q (defaultdict): 训练好的 Q 值表。
        actions (list): 可选动作列表。
        max_steps (int): 最大步数限制，默认为500。

    返回:
        tuple: (步数, 路径)。
            - 步数: 到达目标的步数，若未到达则为 None。
            - 路径: 状态列表 [(row, col), ...]。
    """
    s = env.reset()
    steps = 0
    path = [s]
    while steps < max_steps:
        best_a, best_q = None, float("-inf")
        for a in actions:
            q = Q[(s, a)]
            if q > best_q:
                best_q, best_a = q, a
        if best_a is None:
            best_a = random.choice(actions)
        s, _, done = env.step(best_a)
        path.append(s)
        steps += 1
        if done:
            return steps, path
    return None, path


def visualize_path(env, path, title="Windy Gridworld Path"):
    """
    使用 Pygame 可视化智能体的移动路径。

    参数:
        env (WindyGridworld): 风力格子世界环境。
        path (list): 路径，状态列表 [(row, col), ...]。
        title (str): 窗口标题，默认为"Windy Gridworld Path"。
    """
    # 初始化 Pygame
    pygame.init()

    # 常量定义
    CELL_SIZE = 60  # 单元格大小
    MARGIN = 2  # 单元格间距
    WINDOW_WIDTH = env.n_col * (CELL_SIZE + MARGIN) + MARGIN
    WINDOW_HEIGHT = env.n_row * (CELL_SIZE + MARGIN) + MARGIN + 100  # 额外空间显示信息

    # 颜色定义
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)
    LIGHT_BLUE = (173, 216, 230)
    YELLOW = (255, 255, 0)

    # 创建窗口
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(title)

    # 字体设置
    font = pygame.font.SysFont('方正粗黑宋简体', 24)
    small_font = pygame.font.SysFont('方正粗黑宋简体', 18)

    # 主循环
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 清空屏幕
        screen.fill(WHITE)

        # 绘制网格
        for row in range(env.n_row):
            for col in range(env.n_col):
                x = col * (CELL_SIZE + MARGIN) + MARGIN
                y = row * (CELL_SIZE + MARGIN) + MARGIN

                # 默认单元格颜色
                color = WHITE

                # 起点颜色
                if (row, col) == env.start:
                    color = GREEN

                # 目标点颜色
                elif (row, col) == env.end:
                    color = RED

                # 绘制单元格
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)

                # 绘制风力强度
                if col < len(env.wind) and env.wind[col] > 0:
                    wind_text = small_font.render(f"↑{env.wind[col]}", True, BLUE)
                    text_rect = wind_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE - 10))
                    screen.blit(wind_text, text_rect)

        # 绘制路径
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i + 1]

                x1 = c1 * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2
                y1 = r1 * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2
                x2 = c2 * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2
                y2 = r2 * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2

                # 绘制路径线
                pygame.draw.line(screen, YELLOW, (x1, y1), (x2, y2), 3)

                # 绘制箭头
                if x1 != x2 or y1 != y2:
                    import math
                    angle = math.atan2(y2 - y1, x2 - x1)
                    arrow_length = 10
                    arrow_angle = 0.5

                    x3 = x2 - arrow_length * math.cos(angle - arrow_angle)
                    y3 = y2 - arrow_length * math.sin(angle - arrow_angle)
                    x4 = x2 - arrow_length * math.cos(angle + arrow_angle)
                    y4 = y2 - arrow_length * math.sin(angle + arrow_angle)

                    pygame.draw.polygon(screen, YELLOW, [(x2, y2), (x3, y3), (x4, y4)])

        # 绘制路径点
        for i, (r, c) in enumerate(path):
            x = c * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2
            y = r * (CELL_SIZE + MARGIN) + MARGIN + CELL_SIZE // 2

            # 绘制圆形标记和步数
            pygame.draw.circle(screen, LIGHT_BLUE, (x, y), 15)
            pygame.draw.circle(screen, BLACK, (x, y), 15, 2)

            # 绘制步数编号
            step_text = small_font.render(str(i), True, BLACK)
            text_rect = step_text.get_rect(center=(x, y))
            screen.blit(step_text, text_rect)

        # 绘制信息
        info_y = env.n_row * (CELL_SIZE + MARGIN) + MARGIN + 10

        # 标题
        title_text = font.render(f"路径长度: {len(path) - 1} 步", True, BLACK)
        screen.blit(title_text, (10, info_y))

        # 图例
        legend_y = info_y + 30
        legend_items = [
            ("起点", GREEN),
            ("目标", RED),
            ("路径", YELLOW),
            ("风力 ↑", BLUE)
        ]

        for i, (label, color) in enumerate(legend_items):
            x = 10 + i * 120
            pygame.draw.rect(screen, color, (x, legend_y, 20, 20))
            pygame.draw.rect(screen, BLACK, (x, legend_y, 20, 20), 2)
            label_text = small_font.render(label, True, BLACK)
            screen.blit(label_text, (x + 25, legend_y + 2))

        # 操作提示
        inst_text = small_font.render("按 ESC 或关闭窗口退出", True, GRAY)
        screen.blit(inst_text, (10, legend_y + 30))

        # 更新显示
        pygame.display.flip()
        clock.tick(60)

        # 检查 ESC 键
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

    pygame.quit()


if __name__ == "__main__":
    """
    主程序，运行 SARSA 算法并可视化不同动作集的路径。
    """
    # 四方向动作集
    env = WindyGridworld()
    A4 = env.actions4()
    Q4 = saras(env, A4, episodes=20000)
    best4, path4 = greedy_rollout(env, Q4, A4)
    print(f"找到路径，步数: {best4}，路径: {path4}")
    visualize_path(env, path4, "SARSA - 4 方向动作")

    # 八方向动作集
    env = WindyGridworld()
    A8 = env.actions8()
    Q8 = saras(env, A8, episodes=20000)
    best8, path8 = greedy_rollout(env, Q8, A8)
    print(f"找到路径，步数: {best8}，路径: {path8}")
    visualize_path(env, path8, "SARSA - 8 方向动作")

    # 九方向动作集
    env = WindyGridworld()
    A9 = env.actions9()
    Q9 = saras(env, A9, episodes=20000)
    best9, path9 = greedy_rollout(env, Q9, A9)
    print(f"找到路径，步数: {best9}，路径: {path9}")
    visualize_path(env, path9, "SARSA - 9 方向动作")

    # 随机风力，九方向动作集
    env_random_windy = WindyGridworld(random_windy=True)
    A9 = env_random_windy.actions9()
    Q9 = saras(env_random_windy, A9, episodes=20000)
    best9, path9 = greedy_rollout(env_random_windy, Q9, A9)
    print(f"找到路径，步数: {best9}，路径: {path9}")
    visualize_path(env_random_windy, path9, "SARSA - 9 方向动作 - 随机风力")