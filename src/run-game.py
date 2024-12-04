from stable_baselines3 import DQN
from game import Connect4Env
from random_games import graph_results
from random_games import random_move
from random_vs_model import play_against_random

env = Connect4Env()
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard_logs/")
model.learn(total_timesteps=10_000)

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs)

games_list = play_against_random(env, model, n_games=1000)
model_name = 'test'
graph_results(games_list, player_1_name="DQN Model", player_2_name="Random Agent", save_path=f'result_graphs/{model_name}.png')