import numpy as np
from game import Connect4Env
import matplotlib.pyplot as plt
from collections import Counter

def single_random_game():
    env = Connect4Env()
    obs = env.reset()
    done = False

    while not done:
        valid_actions = [col for col in range(env.action_space.n) if env._is_valid_action(col)]
        if not valid_actions:
            return 0

        action = np.random.choice(valid_actions)
        obs, reward, done, info = env.step(action)

    if reward == 1:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1
    

def graph_results(games_list):
    result_counts = Counter(games_list)

    labels = ['Player 1 Wins', 'Player 2 Wins', 'Draws']
    sizes = [result_counts[1], result_counts[-1], result_counts[0]]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    explode = (0.1, 0.1, 0)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
    plt.title('Game Results Distribution')
    plt.show()

games = []
for i in range(1_000_000):
    result = single_random_game()
    games.append(result)

graph_results(games)
