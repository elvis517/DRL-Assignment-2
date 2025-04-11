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

# from trainfast import NTupleApproximator ,patterns, load_weights
# from policytrain import PolicyApproximator, load_policy_weights
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
def identity(r, c): return r, c
def rot90(r, c): return c, 3 - r
def rot180(r, c): return 3 - r, 3 - c
def rot270(r, c): return 3 - c, r
def flip_h(r, c): return r, 3 - c
def flip_v(r, c): return 3 - r, c
def flip_diag1(r, c): return c, r
def flip_diag2(r, c): return 3 - c, 3 - r
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=5, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        # Q: average reward of the child
        # c: exploration constant (hyperparameter, balances exploration vs. exploitation)
        # UCT encourages trying less-visited nodes if their value is uncertain.
        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            q_value = child.total_reward / child.visits if child.visits > 0 else 0
            uct = q_value + self.c * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        # NOTE: rollout_depth is a hyperparameter — higher depth means more accurate but slower rollouts.
        total_reward = 0
        discount = 1.0

        for _ in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            next_state, new_score, done, _ = sim_env.step(action)
            reward = new_score - sim_env.score
            total_reward += discount * reward
            discount *= self.gamma  # gamma is a hyperparameter (discount factor for future rewards)
            if done:
                return total_reward
        # Instead of running all the way to terminal, we stop early and use the approximator.
        estimated_value = self.approximator.value(sim_env.board)
        return total_reward + discount * estimated_value

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        # As we go back up the tree, we apply discounting by gamma.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma  # gamma is a hyperparameter (discount factor)
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            _, new_score, done, _ = sim_env.step(node.action)
            if done:
                return

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            action = node.untried_actions.pop()
            next_state, new_score, done, _ = sim_env.step(action)
            child_node = TD_MCTS_Node(state=next_state, score=new_score, parent=node, action=action)
            node.children[action] = child_node
            node = child_node  # continue from the newly expanded node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)


    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


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

        self.symmetry_map = list(set(self.symmetry_map))  # 去除重複的對稱映射
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
approximator = NTupleApproximator(board_size=4, patterns=patterns)
def load_weights(approximator, filename_prefix):

    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)
    for j in range(len(weights_data)):
        approximator.weights[j] = defaultdict(lambda: 0, weights_data[j])
    # print(f"📥 Weights loaded from {filename_prefix}.pkl")
load_weights(approximator, "ntuple_1stagefastfastold_whole")


env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=60, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

state = env.reset()



def get_action(state, score):
    root = TD_MCTS_Node(state, score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act





