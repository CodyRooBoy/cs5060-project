from stable_baselines3 import DQN

from game import Connect4Env

env = Connect4Env()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs)
