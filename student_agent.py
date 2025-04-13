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
        self.symmetry_map = list(set(self.symmetry_map))  # 去除重複的對稱映射
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
# PUCT-MCTS
# -------------------------------
class PUCTNode:
    def __init__(self, env, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = env.board.copy()
        self.score = env.score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class MCTS_PUCT:
    def __init__(self, env, approximator, iterations=500, c_puct=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = c_puct
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        return max(
            node.children.values(),
            key=lambda child: self.approximator.value(child.state) + self.c * math.sqrt(math.log(node.visits) / child.visits)
        )

    def rollout(self, sim_env, depth):
        # Perform a random rollout until reaching the maximum depth or a terminal state.
        # Use the approximator to evaluate the final state.
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if legal_actions == []:
                break
            action = random.choice(legal_actions)
            sim_env.step(action, add_tile=False)
        return self.approximator.value(sim_env.board)
        
    def backpropagate(self, node, reward):
        # Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent
            reward *= self.gamma

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions != []:
            action = node.untried_actions.pop()
            sim_env.step(action, add_tile=False)
            node.children[action] = PUCTNode(sim_env, parent=node, action=action)
            node = node.children[action]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        for action, child in root.children.items():
            distribution[action] += child.visits / total_visits if total_visits > 0 else 0
        return np.argmax(distribution), distribution


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
# load_weights(approximator, "/content/drive/MyDrive/DRL/ntuple_1stagefastfastold_whole")

load_weights(approximator, "ntuple_1stagefastfastold_whole")



def get_action(state, score):
    global approximator
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    mcts = MCTS_PUCT(env, approximator, iterations=50, c_puct=1.41, rollout_depth=2, gamma=0.99)
    root = PUCTNode(env)
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
# #重複十次
# for i in range(10):
#   done = False
#   while not done:
#       action = get_action(state, score)
#       state, score, done, _ = env.step(action)
#       # env.render()
#       print(f"Step {step_count+1} | Action: {env.actions[action]} | Score: {score}")
#       step_count += 1

#   print(f"🏁 Game over! Total steps: {step_count}, Final Score: {score}")