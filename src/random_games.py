import numpy as np
from game import Connect4Env
import matplotlib.pyplot as plt
from collections import Counter
import os

def random_move(env):
    valid_actions = [col for col in range(env.action_space.n) if env._is_valid_action(col)]
    if not valid_actions:
        return None
    return np.random.choice(valid_actions)

def single_random_game():
    env = Connect4Env()
    obs = env.reset()
    done = False

    while not done:
        action = random_move(env)
        if action is None:
            return 0

        obs, reward, done, info = env.step(action)

    if reward == 1:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1
    

def graph_results(games_list, player_1_name, player_2_name, save_path=None, verbosity=0):
    result_counts = Counter(games_list)

    labels = [f'{player_1_name} Wins', f'{player_2_name} Wins', 'Draws']
    sizes = [result_counts[1], result_counts[-1], result_counts[0]]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    explode = (0.1, 0.1, 0)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
    plt.title('Game Results Distribution')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        if verbosity == 1:
            print(f"Graph saved to {save_path}")
    else:
        plt.show()

    plt.close()

def play_x_random_games(x):
    games = []
    for i in range(x):
        result = single_random_game()
        games.append(result)

    graph_results(games, 'Player 1', 'Player 2')
