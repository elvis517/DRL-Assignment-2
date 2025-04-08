import numpy as np
import random
import gym
from gym import spaces
# import matplotlib.pyplot as plt

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

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def clone(self):
        clone_env = Game2048Env()
        clone_env.board = self.board.copy()
        clone_env.score = self.score
        return clone_env

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    @staticmethod
    def move_left(board):
        moved = False
        score = 0
        new_board = board.copy()
        for i in range(4):
            row = new_board[i]
            original = row.copy()
            row = row[row != 0]
            row = np.pad(row, (0, 4 - len(row)), 'constant')
            for j in range(3):
                if row[j] != 0 and row[j] == row[j+1]:
                    row[j] *= 2
                    score += row[j]
                    row[j+1] = 0
            row = row[row != 0]
            row = np.pad(row, (0, 4 - len(row)), 'constant')
            new_board[i] = row
            if not np.array_equal(original, row):
                moved = True
        return new_board, moved, score

    @staticmethod
    def move_right(board):
        return Game2048Env.move_left(np.fliplr(board))[::-1], True, 0

    @staticmethod
    def move_up(board):
        transposed, moved, score = Game2048Env.move_left(board.T)
        return transposed.T, moved, score

    @staticmethod
    def move_down(board):
        flipped = np.fliplr(board.T)
        new_board, moved, score = Game2048Env.move_left(flipped)
        return np.fliplr(new_board).T, moved, score

    def step(self, action):
        move_funcs = [self.move_up, self.move_down, self.move_left, self.move_right]
        new_board, moved, gained = move_funcs[action](self.board)
        if moved:
            self.board = new_board
            self.score += gained
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    # def render(self, mode="human", action=None):
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlim(-0.5, self.size - 0.5)
    #     ax.set_ylim(-0.5, self.size - 0.5)
    #     for i in range(self.size):
    #         for j in range(self.size):
    #             value = self.board[i, j]
    #             color = COLOR_MAP.get(value, "#3c3a32")
    #             text_color = TEXT_COLOR.get(value, "white")
    #             rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
    #             ax.add_patch(rect)
    #             if value != 0:
    #                 ax.text(j, i, str(value), ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
    #     title = f"score: {self.score}"
    #     if action is not None:
    #         title += f" | action: {self.actions[action]}"
    #     plt.title(title)
    #     plt.gca().invert_yaxis()
    #     plt.show()

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def get_moves(self):
        moves = []
        for a in range(4):
            new_board, moved, _ = [self.move_up, self.move_down, self.move_left, self.move_right][a](self.board)
            if moved:
                moves.append(a)
        return moves

    def get_afterstate(self, board, action):
        return [self.move_up, self.move_down, self.move_left, self.move_right][action](board)[0]

    def do_move(self, action):
        new_board, moved, gained = [self.move_up, self.move_down, self.move_left, self.move_right][action](self.board)
        if moved:
            self.board = new_board
            self.score += gained
            if np.any(self.board == 0):
                self.add_random_tile()
        return self.is_game_over()
