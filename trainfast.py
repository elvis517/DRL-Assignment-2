import pickle
import math
import numpy as np
import os
from collections import defaultdict
from env2048 import Game2048Env
import random

# -------------------------------
# Transformation functions
# -------------------------------
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

# -------------------------------
# Multi-Stage Approximator
# -------------------------------
class MultiStageApproximator:
    def __init__(self, board_size, patterns, num_stages=3, optimistic_init_value=32000.0):
        self.num_stages = num_stages
        self.stages = [
            NTupleApproximator(board_size, patterns, optimistic_init_value)
            for _ in range(num_stages)
        ]

    def get_stage_index(self, board):
        max_tile = np.max(board)
        if max_tile < 8192:
            return 0
        elif max_tile < 32768:
            return 1
        else:
            return 2

    def value(self, board):
        stage = self.get_stage_index(board)
        return self.stages[stage].value(board)

    def update(self, board, delta, alpha):
        stage = self.get_stage_index(board)
        self.stages[stage].update(board, delta, alpha)

# -------------------------------
# Save and Load
# -------------------------------
def save_weights(approximator, filename_prefix):
    if isinstance(approximator, MultiStageApproximator):
        for i, stage in enumerate(approximator.stages):
            weights_data = [dict(w) for w in stage.weights]
            with open(f"{filename_prefix}_stage{i}.pkl", "wb") as f:
                pickle.dump(weights_data, f)
            print(f"✅ Stage {i} weights saved to {filename_prefix}_stage{i}.pkl")
    else:
        weights_data = [dict(w) for w in approximator.weights]
        with open(f"{filename_prefix}.pkl", "wb") as f:
            pickle.dump(weights_data, f)
        print(f"✅ Weights saved to {filename_prefix}.pkl")

def load_weights(approximator, filename_prefix):
    if isinstance(approximator, MultiStageApproximator):
        for i, stage in enumerate(approximator.stages):
            with open(f"{filename_prefix}_stage{i}.pkl", "rb") as f:
                weights_data = pickle.load(f)
            for j in range(len(weights_data)):
                stage.weights[j] = defaultdict(lambda: 0, weights_data[j])
            print(f"📥 Stage {i} weights loaded from {filename_prefix}_stage{i}.pkl")
    else:
        with open(f"{filename_prefix}.pkl", "rb") as f:
            weights_data = pickle.load(f)
        for j in range(len(weights_data)):
            approximator.weights[j] = defaultdict(lambda: 0, weights_data[j])
        print(f"📥 Weights loaded from {filename_prefix}.pkl")
 
# -------------------------------
# TD Learning Loop
# -------------------------------
def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    final_scores = []
    success_flags = []
    print("Training started...")
    for episode in range(num_episodes):
        state = env.reset()
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            move_values = {}
            for move in legal_moves:
                afterstate = env.get_afterstate(state, move)
                move_values[move] = approximator.value(afterstate)

            # Epsilon-greedy
            # if np.random.rand() < epsilon:
            #     action = np.random.choice(legal_moves)
            # else:
            action = max(move_values, key=move_values.get)

            current_afterstate = env.get_afterstate(state, action)
            current_value = approximator.value(current_afterstate)

            next_state, new_score, done, _ = env.step(action)
            r = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            if not done:
                next_legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                next_value = max([
                    approximator.value(env.get_afterstate(next_state, move))
                    for move in next_legal_moves
                ]) if next_legal_moves else 0
            else:
                next_value = 0

            target = r + gamma * next_value
            delta = target - current_value
            approximator.update(current_afterstate, delta, alpha)

            state = next_state

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        if (episode + 1) % 500 == 0:
            save_weights(approximator, weight_prefix)
            print(f"Checkpoint saved at episode {episode + 1}")

#     return final_scores # # -------------------------------
# def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, batch_size=1):
#     final_scores = []
#     success_flags = []
#     replay_buffer = []  # 存放經驗：(afterstate, reward, next_value)

#     for episode in range(num_episodes):
#         state = env.reset()
#         previous_score = 0
#         done = False
#         max_tile = np.max(state)

#         while not done:
#             legal_moves = [a for a in range(4) if env.is_move_legal(a)]
#             if not legal_moves:
#                 break

#             # 選擇動作 (這裡依然使用貪婪策略)
#             move_values = {}
#             for move in legal_moves:
#                 afterstate = env.get_afterstate(state, move)
#                 move_values[move] = approximator.value(afterstate)
#             action = max(move_values, key=move_values.get)

#             # 當前狀態的 afterstate 與其估值
#             current_afterstate = env.get_afterstate(state, action)
#             current_value = approximator.value(current_afterstate)

#             # 執行動作並獲得下一狀態、得分等
#             next_state, new_score, done, _ = env.step(action)
#             r = new_score - previous_score
#             previous_score = new_score
#             max_tile = max(max_tile, np.max(next_state))

#             # 計算下一狀態的最佳 afterstate 的估值
#             if not done:
#                 next_legal_moves = [a for a in range(4) if env.is_move_legal(a)]
#                 if next_legal_moves:
#                     next_values = [approximator.value(env.get_afterstate(next_state, move))
#                                    for move in next_legal_moves]
#                     next_value = max(next_values)
#                 else:
#                     next_value = 0
#             else:
#                 next_value = 0

#             # 將經驗存入緩衝區
#             replay_buffer.append((current_afterstate, r, next_value))
            
#             # 當累積到一定數量後，使用 numpy 向量化計算 TD 目標與誤差
#             if len(replay_buffer) >= batch_size:
#                 # 將批次資料分離
#                 batch_afterstates, batch_rewards, batch_next_vals = zip(*replay_buffer)
#                 batch_rewards = np.array(batch_rewards)
#                 batch_next_vals = np.array(batch_next_vals)
                
#                 # 計算 TD 目標： target = reward + gamma * next_value
#                 targets = batch_rewards + gamma * batch_next_vals
                
#                 # 對每個 afterstate 用 approximator.value() 計算當前估值
#                 # 注意這裡 approximator.value() 還是無法向量化，因此使用列表推導式
#                 batch_current_vals = np.array([approximator.value(s) for s in batch_afterstates])
                
#                 # 利用 numpy 計算整個批次的 TD 誤差
#                 deltas = targets - batch_current_vals

#                 # 逐一更新每個樣本的權重
#                 for s, delta in zip(batch_afterstates, deltas):
#                     approximator.update(s, delta, alpha)

#                 replay_buffer = []  # 清空緩衝區

#             state = next_state

#         final_scores.append(env.score)
#         success_flags.append(1 if max_tile >= 2048 else 0)

#         if (episode + 1) % 100 == 0:
#             avg_score = np.mean(final_scores[-100:])
#             success_rate = np.sum(success_flags[-100:]) / 100
#             print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
#         if (episode + 1) % 500 == 0:

#             save_weights(approximator, weight_prefix)
#             print(f"Checkpoint saved at episode {episode + 1}")

#     return final_scores

# # # -------------------------------
# Expectimax
# -------------------------------
def expectimax_action(env, approximator, depth=2):
    def expectimax(state, depth, is_player):
        if depth == 0:
            return approximator.value(state)

        shadow_env = Game2048Env()
        shadow_env.board = state.copy()

        if is_player:
            values = []
            for action in range(4):
                shadow_env.board = state.copy()
                if not shadow_env.is_move_legal(action):
                    continue
                shadow_env.score = 0
                next_state, _, _, _ = shadow_env.step(action)
                values.append(expectimax(next_state, depth - 1, False))
            return max(values) if values else approximator.value(state)
        else:
            empty = [(r, c) for r in range(4) for c in range(4) if state[r][c] == 0]
            if not empty:
                return approximator.value(state)
            total_value = 0.0
            for r, c in empty:
                for value, prob in [(2, 0.9), (4, 0.1)]:
                    next_state = state.copy()
                    next_state[r][c] = value
                    total_value += prob * expectimax(next_state, depth - 1, True)
            return total_value / len(empty)

    best_score = -float('inf')
    best_action = None
    for action in range(4):
        if not env.is_move_legal(action):
            continue
        saved_board = env.board.copy()
        saved_score = env.score
        next_state, _, _, _ = env.step(action)
        value = expectimax(next_state, depth - 1, False)
        if value > best_score:
            best_score = value
            best_action = action
        env.board = saved_board.copy()
        env.score = saved_score
    return best_action

# -------------------------------
# Evaluation
# -------------------------------
def play_with_expectimax(env, approximator, depth=2):
    state = env.reset()
    done = False
    while not done:
        action = expectimax_action(env, approximator, depth=depth)
        if action is None:
            break
        state, _, done, _ = env.step(action)
    print(f"🎯 Game Over | Final Score: {env.score} | Max Tile: {np.max(state)}")
    return env.score, np.max(state)

# -------------------------------
# Pattern 設定（Matsuzaki）
# -------------------------------
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

# -------------------------------
# Main Entry
# -------------------------------
USE_MULTISTAGE = False  # 切換 True 為 3-stage，False 為 1-stage

if USE_MULTISTAGE:
    approximator = MultiStageApproximator(board_size=4, patterns=patterns, num_stages=3, optimistic_init_value=0)
    weight_prefix = "ntuple_3stage"
else:
    approximator = NTupleApproximator(board_size=4, patterns=patterns, optimistic_init_value=0)
    weight_prefix = "ntuple_1stagefastfastold_whole"

env = Game2048Env()
# load_weights(approximator, "ntuple_1stagefastfastold_whole")

# final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.005, gamma=0.99)

# save_weights(approximator, weight_prefix)

# 評估範例（可選）
# for _ in range(5):
#     play_with_expectimax(env, approximator, depth=3)
