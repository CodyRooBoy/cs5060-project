import os
from stable_baselines3 import DQN
from game import Connect4Env
from random_vs_model import play_against_random
from random_games import graph_results
from itertools import product
import torch

def generate_model_name(params):
    return "_".join([f"{key}={str(value).replace('.', '')}" for key, value in params.items()])


def train_and_evaluate_grid_search(param_combinations, param_names, env, n_eval_games=100, results_dir="result_graphs", repetitions=3, verbosity=0):
    results = []

    for i, param_set in enumerate(param_combinations):
        params = dict(zip(param_names, param_set))

        for rep in range(repetitions):
            model_name = generate_model_name(params) + f"_rep={rep}"
            if verbosity == 1:
                print(f"Training model {model_name} with parameters: {params}")

            policy_kwargs = dict(
                net_arch=[256, 256, 256],
                activation_fn=torch.nn.ReLU
            )

            model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=verbosity, tensorboard_log=f"./dqn_tensorboard_logs/{model_name}/", **params)
            model.learn(total_timesteps=100_000)
            
            games_list = play_against_random(env, model, n_games=n_eval_games)

            os.makedirs(results_dir, exist_ok=True)
            graph_results(games_list, player_1_name="DQN Model", player_2_name="Random Agent", save_path=f"{results_dir}/{model_name}.png", verbosity=verbosity)
            
            win_rate = games_list.count(1) / len(games_list) * 100
            loss_rate = games_list.count(-1) / len(games_list) * 100
            draw_rate = games_list.count(0) / len(games_list) * 100
            
            results.append({"model_name": model_name, "hyperparameters": params, "win_rate": win_rate, "loss_rate": loss_rate, "draw_rate": draw_rate})
            print(f"Model {model_name} evaluation completed: Wins: {win_rate:.2f}%, Losses: {loss_rate:.2f}%, Draws: {draw_rate:.2f}%")
    
    return results


env = Connect4Env()

param_grid = {
    "learning_rate": [0.001, 0.0005, 0.01],
    "gamma": [0.99, 0.98, 0.9],
    "batch_size": [32, 64],
}


param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

env = Connect4Env()

results = train_and_evaluate_grid_search(param_combinations, param_names, env, n_eval_games=100, results_dir="result_graphs", repetitions=1, verbosity=0)

