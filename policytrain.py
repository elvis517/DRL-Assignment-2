import copy
import math
import random
import numpy as np
from collections import defaultdict
import pickle
import os
from env2048 import Game2048Env
from trainfast import NTupleApproximator
# TODO: Define the action transformation functions (i.e., rot90_action, rot180_action, etc.)
# Note: You have already defined transformation functions for patterns before.
env = Game2048Env()
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

filename_prefix = "policy_weight"

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
print("sat")
def load_weights(approximator, filename_prefix):

    with open(f"{filename_prefix}.pkl", "rb") as f:
        weights_data = pickle.load(f)
    for j in range(len(weights_data)):
        approximator.weights[j] = defaultdict(lambda: 0, weights_data[j])
    print(f"📥 Weights loaded from {filename_prefix}.pkl")
load_weights(approximator, "ntuple_1stagefastfastold_whole")



policy_approximator = PolicyApproximator(board_size=4, patterns=patterns)
load_policy_weights(policy_approximator, filename_prefix)
print("sat")
td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=8, gamma=0.99)
print("start training")
self_play_training_policy_with_td_mcts(env, td_mcts, policy_approximator, num_episodes=500)



