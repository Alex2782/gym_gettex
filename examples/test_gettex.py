import os
import sys
sys.path.append('./') # optional (if not installed via 'pip' -> ModuleNotFoundError)
import gymnasium as gym
import gym_gettex

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO 


# -------------------------------------------------
# load_dict_data
# -------------------------------------------------
def load_dict_data(pickle_path = '../finanzen.net.pickle'):
    
    ret = None
    with open(pickle_path, 'rb') as f_in:
        ret = pickle.loads(f_in.read())

    return ret

# -------------------------------------------------------------------------------------
# INIT Env.
# -------------------------------------------------------------------------------------
env_name = 'GettexStocks-v0'

#isin_list = []
#isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]

# https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
# create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
pickle_path = f'/Users/alex/Develop/gettex/finanzen.net.pickle'
isin_list = load_dict_data(pickle_path)['AKTIE']['isin_list']

date = None

np.random.shuffle(isin_list)

_MAX_DATA = 1 #20 # or None for all data
if _MAX_DATA is not None and len(isin_list) > _MAX_DATA: isin_list = isin_list[:_MAX_DATA]

window_size = 30 #15
prediction_offset = 4 #1

df_list = []
skip_counter = 0
for isin in isin_list:
    
    filename = f'{isin}.csv'
    if date is not None: filename = f'{isin}.{date}.csv'
    path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

    if not os.path.exists(path): 
        skip_counter += 1
        continue

    df = pd.read_csv(path)
    #print ('path:', path)
    start_index = window_size
    #end_index = start_index + 10
    end_index = len(df)
    #end_index = len(df) - 300
    df_dict = dict(df=df, frame_bound = (start_index, end_index))
    df_list.append(df_dict)

print (f'{skip_counter} files skipped')

env = gym.make(env_name,    
    render_mode = None, #"human",
    df = df_list, #df,
    prediction_offset = prediction_offset,
    window_size = window_size,
    frame_bound = (start_index, end_index)
)

idx = 0
for f in env.signal_features:
    #print(f'{idx:>6d} = {f[0]:.3f}, {f[1]:>7.3f}, {f[2]:>7.3f}, {f[3]:>7.3f}, {f[4]:>7.3f}, {f[5]:>7.3f}')
    #print(f'{idx:>6d} = {f:.3f}')
    idx += 1

orig_env = env

env = DummyVecEnv([lambda: env])
env = VecNormalize.load(f'./{env_name}.vec_normalize.pkl', env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

total_num_episodes = len(isin_list) #10

model_path = f'./checkpoint/GettexStocks-v0-149850K-41.PPO_1'
model = PPO.load(model_path, env=env)

print ('model:', model, 'path:', model_path)

vec_env = model.get_env()  
reward_over_episodes = []

tbar = tqdm(range(total_num_episodes))

action_sum = 0
result_counter = {'Action-Sum':0, 'Long-True':0, 'Short-True':0, 'Long-False':0, 'Short-False':0, 'Hold-True':0, 'Hold-False':0}

for episode in tbar:

    obs = vec_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        action_sum += action

        if reward > 0:
            if action > 0: result_counter['Long-True'] += 1
            elif action < 0: result_counter['Short-True'] += 1
        elif reward < 0:
            if action > 0: result_counter['Long-False'] += 1
            elif action < 0: result_counter['Short-False'] += 1
        else:
            if action == 0: result_counter['Hold-True'] += 1
            elif action != 0: result_counter['Hold-False'] += 1            

    reward_over_episodes.append(info[0]['total_reward'])

tbar.close()


print ('Avg. Reward   : ', np.mean(reward_over_episodes))
print ('median Reward : ', np.median(reward_over_episodes))
print ('Min  Reward   : ', np.min(reward_over_episodes))
print ('Max. Reward   : ', np.max(reward_over_episodes))

result_counter['Action-Sum'] = round(float(action_sum[0]), 3)

print(result_counter)

total_profit = info[0]['total_profit']
total_profit_long = info[0]['total_profit_long']
total_profit_short = info[0]['total_profit_short']
print (f'total_profit     : {total_profit:.3f}')
print (f'total_profit_long: {total_profit_long:.3f}')
print (f'total_profit_short: {total_profit_short:.3f}')
print('-'*20)
if len(isin_list) < 10: print (isin_list)
print ('prices:', orig_env.prices[:10])
