import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
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
