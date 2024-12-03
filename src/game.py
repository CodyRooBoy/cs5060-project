import numpy as np
import gym
from gym import spaces

class Connect4Env(gym.Env):
    def __init__(self):
        super(Connect4Env, self).__init__()
        self.board = np.zeros((6, 7), dtype=int)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        if not self._is_valid_action(action):
            return self.board, -10, True, {}
        
        self._drop_token(action)
        reward, done = self._evaluate_game()
        if not done:
            self.current_player = 3 - self.current_player
        return self.board, reward, done, {}

    def _is_valid_action(self, action):
        return self.board[0][action] == 0

    def _drop_token(self, column):
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = self.current_player
                break

    def check_win(self, board, player):
        rows, cols = len(board), len(board[0])
        for row in range(rows):
            for col in range(cols - 3):
                if all(board[row][col + i] == player for i in range(4)):
                    return True

        for col in range(cols):
            for row in range(rows - 3):
                if all(board[row + i][col] == player for i in range(4)):
                    return True

        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(board[row + i][col + i] == player for i in range(4)):
                    return True

        for row in range(3, rows):
            for col in range(cols - 3):
                if all(board[row - i][col + i] == player for i in range(4)):
                    return True

        return False
    
    def check_draw(self, board):
        return all(cell != 0 for row in board for cell in row)
    
    def _evaluate_game(self):
        if self.check_win(self.board, self.current_player):
            return 1 if self.current_player == 1 else -1, True
        elif self.check_draw(self.board):
            return 0, True
        return 0, False