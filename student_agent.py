# TD-Value + MCTS-PUCT 2048 Agent
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
from collections import defaultdict

# -------------------------------
# Game2048 Environment
# -------------------------------
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {k: "#f9f6f2" for k in COLOR_MAP}
TEXT_COLOR[2] = TEXT_COLOR[4] = "#776e65"

def identity(r, c): return r, c

def rot90(r, c): return c, 3 - r

def rot180(r, c): return 3 - r, 3 - c

def rot270(r, c): return 3 - c, r

def flip_h(r, c): return r, 3 - c

def flip_v(r, c): return 3 - r, c

def flip_diag1(r, c): return c, r

def flip_diag2(r, c): return 3 - c, 3 - r

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # 行動空間：0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size-1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size-1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def step(self, action, add_tile=True):
        """執行一個動作，可選擇是否加入隨機 tile（用於 afterstate 模擬）"""
        assert self.action_space.contains(action), "Invalid action"
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False
        self.last_move_valid = moved
        if moved and add_tile:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board.copy(), self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
        title = f"Score: {self.score}"
        if action is not None:
            title += f" | Action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def is_move_legal(self, action):
        temp = self.board.copy()
        if action == 0:
            for j in range(self.size):
                col = temp[:, j]
                new_col = self.compress(col)
                new_col = self.merge(new_col)
                new_col = self.compress(new_col)
                temp[:, j] = new_col
        elif action == 1:
            for j in range(self.size):
                col = temp[:, j][::-1]
                new_col = self.compress(col)
                new_col = self.merge(new_col)
                new_col = self.compress(new_col)
                temp[:, j] = new_col[::-1]
        elif action == 2:
            for i in range(self.size):
                row = temp[i]
                temp[i] = self.compress(row)
                temp[i] = self.merge(temp[i])
                temp[i] = self.compress(temp[i])
        elif action == 3:
            for i in range(self.size):
                row = temp[i][::-1]
                new_row = self.compress(row)
                new_row = self.merge(new_row)
                new_row = self.compress(new_row)
                temp[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp)

    def get_afterstate(self, board, action):
        """
        根據當前 board 與動作，計算 afterstate（不包含隨機新增的 tile）
        """
        board_copy = board.copy()
        def local_compress(row):
            new_row = row[row != 0]
            new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
            return new_row
        def local_merge(row):
            new_row = row.copy()
            for i in range(len(new_row)-1):
                if new_row[i] == new_row[i+1] and new_row[i] != 0:
                    new_row[i] *= 2
                    new_row[i+1] = 0
            return new_row
        if action == 0:
            for j in range(self.size):
                col = board_copy[:, j]
                col = local_compress(col)
                col = local_merge(col)
                col = local_compress(col)
                board_copy[:, j] = col
        elif action == 1:
            for j in range(self.size):
                col = board_copy[:, j][::-1]
                col = local_compress(col)
                col = local_merge(col)
                col = local_compress(col)
                board_copy[:, j] = col[::-1]
        elif action == 2:
            for i in range(self.size):
                row = board_copy[i]
                row = local_compress(row)
                row = local_merge(row)
                row = local_compress(row)
                board_copy[i] = row
        elif action == 3:
            for i in range(self.size):
                row = board_copy[i][::-1]
                row = local_compress(row)
                row = local_merge(row)
                row = local_compress(row)
                board_copy[i] = row[::-1]
        else:
            raise ValueError("Invalid action")
        return board_copy

def get_afterstate(board, action):
    env = Game2048Env()
    env.board = board.copy()
    env.score = 0
    env.step(action, add_tile=False)
    return env.board.copy()

# -------------------------------
# Value Approximator (NTuple)
# -------------------------------
class NTupleApproximator:
    def __init__(self, board_size, patterns, optimistic_init_value=0):
        self.patterns = patterns
        self.weights = [defaultdict(lambda: optimistic_init_value) for _ in patterns]
        self.symmetry_map = []
        seen = set()
        transforms = [identity, rot90, rot180, rot270, flip_v, flip_h, flip_diag1, flip_diag2]
        for i, pattern in enumerate(patterns):
            for t in transforms:
                trans = tuple(t(r, c) for r, c in pattern)
                key = tuple(sorted(trans))
                if key not in seen:
                    seen.add(key)
                    self.symmetry_map.append((i, trans))

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def value(self, board):
        total = 0.0
        for i, coords in self.symmetry_map:
            feature = self.get_feature(board, coords)
            total += self.weights[i][feature]
        return total

    def update(self, board, delta, alpha):
        for i, coords in self.symmetry_map:
            feature = self.get_feature(board, coords)
            self.weights[i][feature] += alpha * delta

# -------------------------------
# Policy Approximator
class PolicyApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.actions = [0, 1, 2, 3]

        self.symmetry_patterns = []
        self.symmetry_types = []
        seen = set()

        for pattern in self.patterns:
            syms, types = self.generate_symmetries(pattern)
            for sym, typ in zip(syms, types):
                canonical = tuple(sorted(sym))
                if canonical not in seen:
                    seen.add(canonical)
                    self.symmetry_patterns.append(sym)
                    self.symmetry_types.append(typ)

        self.weights = [defaultdict(lambda: defaultdict(float)) for _ in range(len(self.symmetry_patterns))]

        self.action_transforms = {
            "identity": lambda a: a,
            "rot90": lambda a: [2, 3, 1, 0][a],
            "rot180": lambda a: [1, 0, 3, 2][a],
            "rot270": lambda a: [3, 2, 0, 1][a],
            "flip_h": lambda a: [0, 1, 3, 2][a],
            "flip_v": lambda a: [1, 0, 2, 3][a],
            "flip_diag1": lambda a: a,
            "flip_diag2": lambda a: [1, 0, 3, 2][a],
        }

    def generate_symmetries(self, pattern):
        transforms = [
            ("identity", identity),
            ("rot90", rot90),
            ("rot180", rot180),
            ("rot270", rot270),
            ("flip_h", flip_h),
            ("flip_v", flip_v),
            ("flip_diag1", flip_diag1),
            ("flip_diag2", flip_diag2),
        ]
        symmetries = []
        types = []
        for name, t in transforms:
            transformed = [t(r, c) for (r, c) in pattern]
            symmetries.append(transformed)
            types.append(name)
        return symmetries, types

    def tile_to_index(self, tile):
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def predict(self, board):
        action_scores = np.zeros(4)
        for pattern, weight, sym_type in zip(self.symmetry_patterns, self.weights, self.symmetry_types):
            feature = self.get_feature(board, pattern)
            action_map = self.action_transforms[sym_type]
            for action in self.actions:
                mapped_action = action_map(action)
                action_scores[action] += weight[feature][mapped_action]

        max_score = np.max(action_scores)
        exp_scores = np.exp(action_scores - max_score)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def update(self, board, target_distribution, alpha=0.1):
        predicted = self.predict(board)
        for pattern, weight, sym_type in zip(self.symmetry_patterns, self.weights, self.symmetry_types):
            feature = self.get_feature(board, pattern)
            action_map = self.action_transforms[sym_type]
            for action in self.actions:
                mapped_action = action_map(action)
                weight[feature][mapped_action] += alpha * (target_distribution[action] - predicted[action])

# PUCT-MCTS
# -------------------------------
class PUCTNode:
    def __init__(self, state, score, parent=None, action=None, prior=1.0):
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if self.is_legal_action(state, a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    @staticmethod
    def is_legal_action(state, action):
        env = Game2048Env()
        env.board = state.copy()
        return env.is_move_legal(action)

class MCTS_PUCT:
    def __init__(self, env, value_approximator, policy_approximator=None, iterations=100, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.value_approximator = value_approximator
        self.policy_approximator = policy_approximator  # ← 加這行
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        env = copy.deepcopy(self.env)
        env.board = state.copy()
        env.score = score
        return env

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values()) + 1e-8
        best_score = -float('inf')
        best_child = None
        for action, child in node.children.items():
            q = child.total_reward / (child.visits + 1e-8)
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        """
        policy-guided rollout 到最大 depth，然後用 value approximator 評估。
        如果 depth=0，等於直接使用 value function。
        """
        total_reward = 0.0
        discount = 1.0
        steps = 0

        while steps < depth and not sim_env.is_game_over():
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break

            if self.policy_approximator:
                probs = self.policy_approximator.predict(sim_env.board)
                probs = np.array([probs[a] if a in legal_moves else 0 for a in range(4)])
                probs /= np.sum(probs) + 1e-8
                action = np.random.choice(4, p=probs)
            else:
                action = random.choice(legal_moves)

            _, reward, done, _ = sim_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            steps += 1

            if done:
                return total_reward

        # 掃尾用 value approximator
        value = self.value_approximator.value(sim_env.board)
        return total_reward + discount * value


    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            _, new_score, done, _ = sim_env.step(node.action)
            sim_env.score = new_score
            if done:
                return
        if not node.untried_actions and not node.children:
            return  # 無合法動作可擴展
        
        if not node.fully_expanded():
            action = node.untried_actions.pop()
            afterstate = get_afterstate(sim_env.board, action)
            next_state, new_score, done, _ = sim_env.step(action)

            # 使用 policy approximator
            if self.policy_approximator is not None:
                policy_probs = self.policy_approximator.predict(sim_env.board)
                prior = policy_probs[action]
            else:
                prior = 1.0

            child = PUCTNode(state=afterstate, score=new_score, parent=node, action=action, prior=prior)
            node.children[action] = child
            node = child
            sim_env.board = next_state.copy()
            sim_env.score = new_score


        reward = self.rollout(sim_env)
        self.backpropagate(node, reward)

    def best_action_distribution(self, root):
        total = sum(child.visits for child in root.children.values()) + 1e-8
        best_visits, best_action = -1, None
        distribution = np.zeros(4)
        for action, child in root.children.items():
            distribution[action] = child.visits / total
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

# -------------------------------
# 推論接口：get_action
# -------------------------------

patterns = [
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    [(2,0), (2,1), (2,2), (2,3), (3,0), (3,1)],
    [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
    [(0,0), (0,1), (0,2), (0,3), (1,0), (1,2)],
    [(0,0), (0,1), (1,0), (1,1)],
    [(1,0), (1,1), (2,0), (2,1)],
    [(0,0), (0,1), (0,2), (0,3)],
    [(1,0), (1,1), (1,2), (1,3)]
]
approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init_value=0)
def load_weights(approximator, filename_prefix):
    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)
    for j in range(len(weights_data)):
        approximator.weights[j] = defaultdict(lambda: 0, weights_data[j])
load_weights(approximator, "ntuple_1stagefastfastold_whole")

policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)
def load_policy_weights(policy_approximator, filename_prefix, default_value=0.0):
    """
    從檔案中讀取 PolicyApproximator 的權重，並還原成 defaultdict 結構。
    權重數量應與 symmetry_patterns 數量一致。
    """
    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)

    if len(weights_data) != len(policy_approximator.symmetry_patterns):
        raise ValueError(
            f"❌ 錯誤：載入的權重數量（{len(weights_data)}）與 symmetry_patterns 不一致（{len(policy_approximator.symmetry_patterns)}）"
        )

    new_weights = []
    for w in weights_data:
        new_w = defaultdict(lambda: defaultdict(lambda: default_value))
        for feature, action_dict in w.items():
            new_w[feature] = defaultdict(float, action_dict)
        new_weights.append(new_w)

    policy_approximator.weights = new_weights
    print(f"📥 Policy weights loaded from {filename_prefix}.pkl")
load_policy_weights(policy_approximator, "policy_weight")


def get_action(state, score):
    global approximator
    global policy_approximator
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    mcts = MCTS_PUCT(env, approximator, iterations=120 , c_puct=1.41, rollout_depth=5, gamma=0.99)
    root = PUCTNode(env.board, env.score)
    for _ in range(mcts.iterations):
        mcts.run_simulation(root)
    action, _ = mcts.best_action_distribution(root)
    return action
# -------------------------------
# # 最終 get_action 函數：使用 PUCT-MCTS 並整合 afterstate
# done = False
# env = Game2048Env()
# state = env.reset()
# score = 0
# step_count = 0

# while not done:
#     action = get_action(state, score)
#     state, score, done, _ = env.step(action)
#     # env.render()
#     print(f"Step {step_count+1} | Action: {env.actions[action]} | Score: {score}")
#     step_count += 1

# print(f"🏁 Game over! Total steps: {step_count}, Final Score: {score}")