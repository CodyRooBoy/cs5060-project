import numpy as np
from game import Connect4Env
import matplotlib.pyplot as plt
from collections import Counter
from random_games import random_move

def play_against_random(env, model, n_games=1000):
    games_list = []
    
    for _ in range(n_games):
        obs = env.reset()
        done = False
        current_player = 1
        
        while not done:
            if current_player == 1:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = random_move(env)
                if action is None:
                    break
            
            obs, reward, done, _ = env.step(action)
            if done:
                games_list.append(reward)
            
            current_player = 3 - current_player
    
    return games_list
