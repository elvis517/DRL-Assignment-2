# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
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

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)



# 假設以下轉換函數 (identity, rot90, rot180, rot270, flip_v, flip_h, flip_diag1, flip_diag2) 已經定義
# 假設 Game2048Env 已經定義並且可以正常運作
# 假設 td_approximator 為訓練好的 NTupleApproximator 的實例
# 例如：
# td_approximator = NTupleApproximator(board_size=4, patterns=your_patterns, optimistic_init_value=0.5)
#
# 下面提供 PUCT-MCTS 節點與搜尋實作
env = Game2048Env()
class PUCTNode:
    def __init__(self, state, score, parent=None, action=None, prior=0.0):
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}  # 格式: {action: child_node}
        self.visits = 0
        self.total_reward = 0.0
        # 利用全域環境 env（或採用 Game2048Env() 的新實例）獲取合法動作
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0

# MCTS_PUCT 類別，結合了 value 與 policy approximator
class MCTS_PUCT:
    def __init__(self, env, value_approximator, policy_approximator, iterations=500, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.value_approximator = value_approximator
        self.policy_approximator = policy_approximator
        self.iterations = iterations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        """建立環境的深拷貝，用於模擬"""
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values()) + 1e-8  # avoid division by zero
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q = child.total_reward / (child.visits + 1e-8)
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            puct_score = q + u
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child
        return best_child, best_action

    def rollout(self, sim_env, depth):
        # 目前直接利用已訓練的 value approximator 評估葉節點
        value_est = self.value_approximator.value(sim_env.board)
        return value_est

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma  # 應用折扣因子
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection Phase：從根節點向下選擇直到遇到可擴展的節點
        while node.fully_expanded() and node.children:
            node, _ = self.select_child(node)
            # 執行選擇的動作模擬狀態更新
            next_state, new_score, done, _ = sim_env.step(node.action) if node.action is not None else (sim_env.board.copy(), sim_env.score, False, {})
            sim_env.board = next_state.copy()
            sim_env.score = new_score
            if done:
                break

        # Expansion Phase：若還有未試過的動作且非終止狀態，擴展節點
        if not node.fully_expanded():
            action = node.untried_actions.pop()
            next_state, new_score, done, _ = sim_env.step(action)

            # 取得該狀態下的先驗分佈，使用 policy approximator
            priors = self.policy_approximator.predict(next_state)
            prior_prob = priors[action]

            child_node = PUCTNode(state=next_state, score=new_score, parent=node, action=action, prior=prior_prob)
            node.children[action] = child_node
            node = child_node

            sim_env.board = next_state.copy()
            sim_env.score = new_score

        # Rollout Phase：從擴展節點進行 rollout 模擬
        rollout_reward = self.rollout(sim_env, self.rollout_depth)

        # Backpropagation Phase：將 rollout 的結果向上反向更新
        self.backpropagate(node, rollout_reward)

    def search(self, root):
        for _ in range(self.iterations):
            self.run_simulation(root)
        return root

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values()) + 1e-8
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

    def best_action(self, root):
        best_action, _ = self.best_action_distribution(root)
        return best_action

# -------------------------------
# 重寫 get_action 函數：結合 value 與 policy
# approximator
def identity(r, c): return r, c
def rot90(r, c): return c, 3 - r
def rot180(r, c): return 3 - r, 3 - c
def rot270(r, c): return 3 - c, r
def flip_h(r, c): return r, 3 - c
def flip_v(r, c): return 3 - r, c
def flip_diag1(r, c): return c, r
def flip_diag2(r, c): return 3 - c, 3 - r

# -------------------------------
# Base NTuple Approximator
# -------------------------------
class NTupleApproximator:
    def __init__(self, board_size, patterns, optimistic_init_value=32000.0):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(lambda: optimistic_init_value) for _ in patterns]
        self.symmetry_map = []
        
        transforms = [identity, rot90, rot180, rot270, flip_v, flip_h, flip_diag1, flip_diag2]
        seen = set()  # 用來記錄已經出現過的規範化 pattern
        
        for i, pattern in enumerate(self.patterns):
            for t in transforms:
                transformed = tuple(t(r, c) for r, c in pattern)
                # 規範化處理：將轉換後的 pattern 排序後作為唯一標識
                canonical = tuple(sorted(transformed))
                if canonical not in seen:
                    seen.add(canonical)
                    self.symmetry_map.append((i, transformed))
        
        # print(self.symmetry_map)

        # # self.symmetry_map = list(set(self.symmetry_map))  # 去除重複的對稱映射
        # print(f"NTupleApproximator initialized with {len(self.weights)} patterns and {len(self.symmetry_map)} symmetry mappings.")

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
class PolicyApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want.
        """
        self.board_size = board_size
        self.patterns = patterns
        self.actions = [0, 1, 2, 3]  # 上下左右

        # Weight structure: [pattern_idx][feature_key][action]
        self.weights = [defaultdict(lambda: defaultdict(float)) for _ in range(len(patterns))]

        # Generate the 8 symmetrical transformations for each pattern and store their types.
        self.symmetry_patterns = []
        self.symmetry_types = []  # Store the type of symmetry transformation (rotation or reflection)
        for pattern in self.patterns:
            syms, types = self.generate_symmetries(pattern)
            self.symmetry_patterns.extend(syms)
            self.symmetry_types.extend(types)

        # TODO: Define corresponding action transformation functions for each symmetry.
        # These are needed to map action probabilities consistently.
        self.action_transforms = {
            "identity": lambda a: a,
            "rot90": lambda a: [2, 3, 1, 0][a],     # L->D->R->U
            "rot180": lambda a: [1, 0, 3, 2][a],    # U<->D, L<->R
            "rot270": lambda a: [3, 2, 0, 1][a],
            "flip_h": lambda a: [0, 1, 3, 2][a],    # 左右對調
            "flip_v": lambda a: [1, 0, 2, 3][a],    # 上下對調
            "flip_diag1": lambda a: [0, 1, 2, 3][a],  # 通常不變
            "flip_diag2": lambda a: [1, 0, 3, 2][a],  # 近似 rot180
        }

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
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
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[r][c]) for (r, c) in coords)

    def predict(self, board):
        # TODO: Predict the policy (probability distribution over actions) given the board state.
        action_scores = np.zeros(4)
        for pattern, weight in zip(self.symmetry_patterns, self.weights * 8):
            feature = self.get_feature(board, pattern)
            for action in self.actions:
                action_scores[action] += weight[feature][action]

        # Convert scores to probability via softmax
        max_score = np.max(action_scores)
        exp_scores = np.exp(action_scores - max_score)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def update(self, board, target_distribution, alpha=0.1):
        # TODO: Update policy based on the target distribution.
        predicted = self.predict(board)
        for pattern, weight in zip(self.symmetry_patterns, self.weights * 8):
            feature = self.get_feature(board, pattern)
            for action in self.actions:
                weight[feature][action] += alpha * (target_distribution[action] - predicted[action])
patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3), 
     (1, 0), (1, 1)],
    [(1, 0), (1, 1), (1, 2), (1, 3), 
     (2, 0), (2, 1)],
    [(2, 0), (2, 1), (2, 2), (2, 3), 
     (3, 0), (3, 1)],

    [(0, 0), (0, 1), (0, 2), 
     (1, 0), (1, 1), (1, 2)],
    [(1, 0), (1, 1), (1, 2), 
     (2, 0), (2, 1), (2, 2)],

    [(0, 0), (0, 1), (0, 2), (0, 3), 
     (1, 0),        (1, 2)],

    [(0, 0), (0, 1), 
     (1, 0), (1, 1)],
    [(1, 0), (1, 1), 
     (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    
]
td_approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init_value=0)

def load_weights(approximator, filename_prefix):

    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)
    for j in range(len(weights_data)):
        approximator.weights[j] = defaultdict(lambda: 0, weights_data[j])
    # print(f"📥 Weights loaded from {filename_prefix}.pkl")
load_weights(td_approximator, "ntuple_1stagefastfastold_whole")
policy_approximator = PolicyApproximator(env, patterns=patterns, weights=None)
def load_policy_weights(policy_approximator, filename_prefix, default_value=0.0):
    """
    從檔案中讀取 PolicyApproximator 的權重，並還原成 defaultdict 的結構。
    這裡每個 feature 的對應權重將以 defaultdict(float) 還原。
    """
    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)
    
    new_weights = []
    for w in weights_data:
        new_w = defaultdict(lambda: defaultdict(lambda: default_value))
        for feature, action_dict in w.items():
            new_w[feature] = defaultdict(float, action_dict)
        new_weights.append(new_w)
    
    policy_approximator.weights = new_weights
    print(f"📥 Policy weights loaded from {filename_prefix}.pkl")
load_policy_weights(policy_approximator, "policy_weight.pkl")

def get_action(state, score):
    """
    根據當前狀態與得分，建立 MCTS 搜索樹，利用已訓練的 TD value approximator 和 policy approximator，
    最終返回最佳動作 (0: up, 1: down, 2: left, 3: right)。
    """
    # 建立一個新的環境實例
    env_inst = copy.deepcopy(env)
    env_inst.board = state.copy()
    env_inst.score = score

    # 建立根節點
    root = PUCTNode(state, score, parent=None, action=None, prior=0.0)
    # 透過 policy approximator 得到根狀態先驗分佈，更新根節點的先驗
    policy_probs = policy_approximator.predict(state)
    # 針對所有根節點的合法動作進行更新
    for a in root.untried_actions:
        root.prior = 0  # 根節點自身的 prior 可以維持 0
        # 若該動作在 policy_prob 中有定義，則設為相應先驗值；否則均勻分配
    # 注意：下方在擴展階段會從 policy 取得先驗，所以此處可以不用顯式地更新 root.P

    # 建立 MCTS 搜索器，同時傳入 value 與 policy approximator
    mcts = MCTS_PUCT(env_inst, td_approximator, policy_approximator, iterations=500, c_puct=1.41, rollout_depth=10, gamma=0.99)
    # 執行 MCTS 搜索
    root = mcts.search(root)
    # 根據搜尋結果選擇訪問次數最多的動作
    best_a = mcts.best_action(root)
    return best_a





