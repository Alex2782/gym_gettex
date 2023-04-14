# stable-baselines3, Add Gymnasium support: 
# https://github.com/DLR-RM/stable-baselines3/pull/1327
# pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support

import os
import pickle
import sys
sys.path.append('./') # optional (if not installed via 'pip' -> ModuleNotFoundError)

import gymnasium as gym
import gym_gettex

import numpy as np
import pandas as pd
from datetime import datetime

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO 

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from copy import deepcopy
from tqdm import tqdm
import random
import torch


def print_stats(reward_over_episodes):
    """print Reward avg. / min. / max. """

    avg = np.mean(reward_over_episodes)
    min = np.min(reward_over_episodes)
    max = np.max(reward_over_episodes)

    print (f'Min. Reward          : {min:>10.3f}')
    print (f'Avg. Reward          : {avg:>10.3f}')
    print (f'Max. Reward          : {max:>10.3f}')

    return min, avg, max

# TRAINING + TEST
# =========================================================
def train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps=10_000, eval_callback=None):
    """ if model=None then execute 'Random actions' """

    #reproduce training and test
    print ('-' * 80)
    #obs = env.reset(seed=seed)
    obs = orig_env.reset(seed=seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    vec_env = None

    if model is not None:
        print(f'model {type(model)}')
        print(f'policy {type(model.policy)}')
        #print(f'model.learn(): {total_learning_timesteps} timesteps ...')

        model.learn(total_timesteps=total_learning_timesteps, progress_bar=True, callback=eval_callback)
        # ImportError: You must install tqdm and rich in order to use the progress bar callback. 
        # It is included if you install stable-baselines with the extra packages: `pip install stable-baselines3[extra]`

        vec_env = model.get_env()
        obs = vec_env.reset()
    else:
        print ("RANDOM actions")

    reward_over_episodes = []

    tbar = tqdm(range(total_num_episodes))

    if vec_env: obs = vec_env.reset()

    for episode in tbar:
        
        if vec_env: 
            #obs = vec_env.reset()
            pass
        else:
            obs, info = orig_env.reset()

        total_reward = 0
        done = False
        while not done:

            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
            else: #random
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = orig_env.step(action)
                done = terminated or truncated 

            total_reward += reward
            if done: break
        
        reward_over_episodes.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_over_episodes)
            tbar.set_description(f'Episode: {episode}, Avg. Reward: {avg_reward:.3f}')
            tbar.update()

    tbar.close()
    avg_reward = np.mean(reward_over_episodes)
    
    return reward_over_episodes

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


window_size = 30 #15
prediction_offset = 4 #1

date = None
#date = '2023-03-29'

np.random.shuffle(isin_list)

_MAX_DATA = 1000 # or None for all data
if _MAX_DATA is not None and len(isin_list) > _MAX_DATA: isin_list = isin_list[:_MAX_DATA]


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
    #print ('df:', df)
    #print ('path:', path)
    start_index = window_size
    end_index = len(df)
    #end_index = len(df) - 100
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
orig_env = env

env = Monitor(env, filename='./tensorboard_log')
env = DummyVecEnv([lambda: env])


# RESUME training?
RESUME = False #True
resume_model = None

if RESUME:

    env = VecNormalize.load(f'./{env_name}.vec_normalize.pkl', env)
    #  do not update them at test time
    env.training = True
    # reward normalization is not needed at test time
    env.norm_reward = True    

    #model_path = f'./checkpoint/GettexStocks-v0-30000K-30.PPO'
    #resume_model = PPO.load(model_path, env=env)
    model_path = f'./checkpoint/GettexStocks-v0-2000K-25.RecurrentPPO'
    resume_model = RecurrentPPO.load(model_path, env=env)

else:
    #print ('obs:', env.reset())
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    #print ('VecNormalize-obs:', env.reset())


#eval_env = deepcopy(env) #TODO EvalCalback, normal erstellen, deepcopy geht nicht

seed = 42 #random seed
total_num_episodes = 50 #len(isin_list) #250
#del isin_list

eval_freq = 100_000

print ("env_name                 :", env_name)
print ("seed                     :", seed)
print ("total_num_episodes       :", total_num_episodes)

# Random actions
model = None 
total_learning_timesteps = 0
rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps)
min, avg, max = print_stats(rewards)
class_name = f'Random actions'
avg = np.mean(rewards)
label = f'Avg. {avg:>7.2f} : {class_name}'


learning_timesteps_list_in_K = [10_000]  # 10k -> PPO = 1:15h, RecurrentPPO = 10:30h, A2C = 1:15h, TRPO = 50m, ARS = 23 min
#learning_timesteps_list_in_K = [12_000] # (RecurrentPPO) = 12h
#learning_timesteps_list_in_K = [45_000] #(for a2c, ppo, trpo) = 15h
#learning_timesteps_list_in_K = [30_000] #(for a2c, ppo, trpo) = 10h
#learning_timesteps_list_in_K = [15_000] #(for a2c, ppo, trpo) = 5h
#learning_timesteps_list_in_K = [500, 1000, 5000, 25_000]
#learning_timesteps_list_in_K = [150_000]

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
model_class_list = [A2C, PPO, A2C, PPO, A2C, PPO, A2C, PPO, A2C, PPO]
#model_class_list = [RecurrentPPO]
#model_class_list = [A2C, PPO, TRPO, ARS]
#model_class_list = [A2C, PPO]
#model_class_list = [A2C, PPO, RecurrentPPO, TRPO]
#model_class_list = [A2C, DDPG, DQN, PPO, SAC, TD3,
#                    ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO] #from sb3_contrib

for timesteps in learning_timesteps_list_in_K:

    total_learning_timesteps = timesteps * 1000
    step_key = f'{timesteps}K'

    for model_class in model_class_list:
        
        policy_dict = model_class.policy_aliases
        #https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        # MlpPolicy or MlpLstmPolicy
        policy = policy_dict.get('MlpPolicy')
        if policy is None: policy = policy_dict.get('MlpLstmPolicy')

        try:
            if resume_model is None:
                model = model_class(policy, env, verbose=0, tensorboard_log="./tensorboard_log/")
            else:
                model = resume_model

            class_name = type(model).__qualname__

            eval_callback = None
            #TODO ERROR: 'NoneType' object has no attribute 'env_is_wrapped'
            #eval_callback = EvalCallback(eval_env, best_model_save_path="./checkpoint/",
            #                            log_path="./tensorboard_log/", eval_freq=eval_freq,
            #                            deterministic=True, render=False)

            rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps, eval_callback)
            min, avg, max, = print_stats(rewards)
            label = f'Avg. {avg:>7.2f} : {class_name} - {step_key}'

            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'./checkpoint/{env_name}-{step_key}-{int(avg)}.{now}.{class_name}'
            model.save(model_path)
                   
        except Exception as e:
            print (f'ERROR: {str(e)}')
            continue

#save the VecNormalize statistics
env.save(f'./{env_name}.vec_normalize.pkl')
