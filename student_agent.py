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
from trainfast import NTupleApproximator ,patterns
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

import copy
import math
import random
import numpy as np

# 假設以下轉換函數 (identity, rot90, rot180, rot270, flip_v, flip_h, flip_diag1, flip_diag2) 已經定義
# 假設 Game2048Env 已經定義並且可以正常運作
# 假設 td_approximator 為訓練好的 NTupleApproximator 的實例
# 例如：
# td_approximator = NTupleApproximator(board_size=4, patterns=your_patterns, optimistic_init_value=0.5)
#
# 下面提供 PUCT-MCTS 節點與搜尋實作

class PUCTNode:
    def __init__(self, state, score, parent=None, action=None, prior=0.0):
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}         # 格式：{action: child_node}
        self.visits = 0
        self.total_reward = 0.0
        # 取得合法動作：這裡使用環境法，建構一個臨時環境以取得當前狀態的合法動作
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        temp_env = Game2048Env()
        temp_env.board = self.state.copy()
        actions = []
        for a in range(temp_env.action_space.n):
            if temp_env.is_move_legal(a):
                actions.append(a)
        return actions

    def fully_expanded(self):
        return len(self.untried_actions) == 0

# PUCT-MCTS 搜索類別
class MCTS_PUCT:
    def __init__(self, env, td_approximator, iterations=500, c_puct=1.41, gamma=0.99):
        self.env = env
        self.td_approximator = td_approximator
        self.iterations = iterations
        self.c_puct = c_puct
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values()) + 1e-8  # 避免除零
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
        # 這裡直接使用 td_approximator 評估當前狀態作為 roll-out 的結果
        value_est = self.td_approximator.value(sim_env.board)
        return value_est

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection phase: traverse the tree until reaching an expandable node.
        while node.fully_expanded() and node.children:
            node, _ = self.select_child(node)
            # 模擬在 sim_env 上執行該節點對應的動作
            next_state, new_score, done, _ = sim_env.step(node.action) if node.action is not None else (sim_env.board.copy(), sim_env.score, False, {})
            sim_env.board = next_state.copy()
            sim_env.score = new_score
            if done:
                break

        # Expansion phase: 如果節點尚未完全展開且非終止狀態，擴展一個未試過的動作
        if not node.fully_expanded():
            action = node.untried_actions.pop()
            next_state, new_score, done, _ = sim_env.step(action)
            # 取得先驗機率：此處假設無政策近似器，先驗可均勻設定
            prior_prob = 1.0 / len(node.get_legal_actions()) if node.get_legal_actions() else 0.0
            child_node = PUCTNode(state=next_state, score=new_score, parent=node, action=action, prior=prior_prob)
            node.children[action] = child_node
            node = child_node
            # 更新 sim_env 狀態
            sim_env.board = next_state.copy()
            sim_env.score = new_score

        # Rollout phase: 從擴展後的節點進行 roll-out 評估
        rollout_reward = self.rollout(sim_env, depth=3)  # depth 可根據需要設定
        # Backpropagation phase: 將 rollout 得到的 reward 反向傳播
        self.backpropagate(node, rollout_reward)

    def search(self, root):
        for _ in range(self.iterations):
            self.run_simulation(root)
        return root

    def best_action(self, root):
        total_visits = sum(child.visits for child in root.children.values()) + 1e-8
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action


def identity(r, c): return r, c
def rot90(r, c): return c, 3 - r
def rot180(r, c): return 3 - r, 3 - c
def rot270(r, c): return 3 - c, r
def flip_h(r, c): return r, 3 - c
def flip_v(r, c): return 3 - r, c
def flip_diag1(r, c): return c, r
def flip_diag2(r, c): return 3 - c, 3 - r


patterns=patterns = [
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
    print(f"📥 Weights loaded from {filename_prefix}.pkl")

load_weights(td_approximator, "ntuple_1stagefastfastold_50000")




# 最終接口：根據當前狀態與分數進行 MCTS 搜索，並返回最佳動作
def get_action(state, score):
    # 使用當前環境實例（可以全域初始化 env）
    env_instance = Game2048Env()
    env_instance.board = state.copy()
    env_instance.score = score
    # 建立根節點
    root = PUCTNode(state, score, parent=None, action=None, prior=0.0)
    # 初始化 MCTS_PUCT 搜索器
    mcts = MCTS_PUCT(env_instance, td_approximator, iterations=500, c_puct=1.41, gamma=0.99)
    # 執行 MCTS 搜索
    root = mcts.search(root)
    # 選擇最佳動作（例如訪問數最多）
    best_a = mcts.best_action(root)
    return best_a





