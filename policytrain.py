import copy
import math
import random
import numpy as np
from collections import defaultdict
import pickle
import os
from env2048 import Game2048Env
from trainfast import NTupleApproximator
from MCTS import TD_MCTS_Node, TD_MCTS
# TODO: Define the action transformation functions (i.e., rot90_action, rot180_action, etc.)
# Note: You have already defined transformation functions for patterns before.
def save_policy_weights(policy_approximator, filename_prefix):
    """
    將 PolicyApproximator 的權重儲存到檔案中。
    權重結構為 [pattern_idx]，每個元素是 defaultdict(lambda: defaultdict(float))。
    此函數會先將每個 defaultdict 轉換為普通字典，再以 pickle 儲存。
    """
    weights_data = []
    for w in policy_approximator.weights:
        # 將每個 feature 屬性轉換成普通字典
        w_dict = {feature: dict(action_dict) for feature, action_dict in w.items()}
        weights_data.append(w_dict)
    
    with open(f"{filename_prefix}.pkl", "wb") as f:
        pickle.dump(weights_data, f)
    print(f"✅ Policy weights saved to {filename_prefix}.pkl")


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


filename_prefix = "policy_weights"

def identity(r, c):
    return r, c

def rot90(r, c):
    return c, 3 - r

def rot180(r, c):
    return 3 - r, 3 - c

def rot270(r, c):
    return 3 - c, r

# 翻轉
def flip_h(r, c):
    return r, 3 - c  # 水平翻轉：左右對調

def flip_v(r, c):
    return 3 - r, c  # 垂直翻轉：上下對調

# 對角翻轉
def flip_diag1(r, c):  # 沿 ↘ 對角線翻轉（主對角線）
    return c, r

def flip_diag2(r, c):  # 沿 ↙ 對角線翻轉（副對角線）
    return 3 - c, 3 - r


# Note: PolicyApproximator is similar to the value approximator but differs in key aspects.
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

def self_play_training_policy_with_td_mcts(env, td_mcts, policy_approximator, num_episodes=50):
    episode_scores = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        print("start")
        while not done:
            # Create the root node for the TD-MCTS tree
            root = TD_MCTS_Node(state, env.score)

            # Run multiple simulations to build the MCTS search tree
            for _ in range(td_mcts.iterations):
                td_mcts.run_simulation(root)

            best_action, target_distribution = td_mcts.best_action_distribution(root)

            # TODO: Update the NTuple Policy Approximator using the MCTS action distribution
            policy_approximator.update(state, target_distribution, alpha=0.01)

            # Execute the selected action in the real environment
            state, reward, done, _ = env.step(best_action)

        # 記錄最終得分
        episode_scores.append(env.score)

        # 顯示進度
        print(f"Episode {episode+1}/{num_episodes} finished, final score: {env.score}")

        if (episode + 1) % 1 == 0:
            avg = np.mean(episode_scores[-1:])
            print(f"📊 [Ep {episode+1}] Avg Score (last 10): {avg:.2f}")
            save_policy_weights(policy_approximator, "policy_weight")
    return episode_scores


env = Game2048Env()

# TODO: Define your own pattern
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
    print(f"📥 Weights loaded from {filename_prefix}.pkl")
load_weights(approximator, "ntuple_1stagefastfastold_50000")



policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)
td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=5, gamma=0.99)
print("start training")
self_play_training_policy_with_td_mcts(env, td_mcts, policy_approximator, num_episodes=500)



