import os
import sys
sys.path.append('./') # optional (if not installed via 'pip' -> ModuleNotFoundError)
import gymnasium as gym
import gym_gettex

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO 



# -------------------------------------------------------------------------------------
# test_model
# -------------------------------------------------------------------------------------
def test_model(window_size=30, prediction_offset=2, max_data=256, isin_list = [], date = None, 
               output_min_profit=0.0, model_path=None, vec_normalize_path=None):

    env_name = 'GettexStocks-v0'

    #isin_list = ['US6701002056', 'US22658D1000', 'US1567271093', 'US0028962076', 'US69370C1009', 'SE0012141687', 'PLBH00000012']
    #isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]

    # https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
    # create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
    if len(isin_list) == 0:
        pickle_path = f'/Users/alex/Develop/gettex/finanzen.net.pickle'
        isin_list = load_dict_data(pickle_path)['AKTIE']['isin_list']

    np.random.shuffle(isin_list)

    if max_data is not None and len(isin_list) > max_data: isin_list = isin_list[:max_data]

    df_list = []

    skip_counter = 0
    for isin in isin_list:
        
        filename = f'{isin}.csv'
        if date is not None: filename = f'{isin}.{date}.csv'
        path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

        if not os.path.exists(path): 
            skip_counter += 1
            continue

        df = pd.read_csv(path, dtype=float)

        #print ('path:', path)
        start_index = window_size
        #end_index = start_index + 10
        end_index = len(df)
        #end_index = len(df) - 300
        df_dict = dict(isin=isin, df=df, frame_bound = (start_index, end_index))
        df_list.append(df_dict)

    print (f'{skip_counter} files skipped')

    env = gym.make(env_name,    
        render_mode = None, #"human",
        df = df_list, #df,
        prediction_offset = prediction_offset,
        window_size = window_size
    )

    idx = 0
    for f in env.signal_features:
        #print(f'{idx:>6d} = {f[0]:.3f}, {f[1]:>7.3f}, {f[2]:>7.3f}, {f[3]:>7.3f}, {f[4]:>7.3f}, {f[5]:>7.3f}')
        #print(f'{idx:>6d} = {f:.3f}')
        idx += 1

    orig_env = env

    env = DummyVecEnv([lambda: env])

    if vec_normalize_path is not None:
        env = VecNormalize.load(vec_normalize_path, env)

    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    total_num_episodes = len(isin_list) - skip_counter #10

    model_class_dict = {'A2C':A2C, 'DDPG':DDPG, 'DQN':DQN, 'PPO':PPO, 'SAC':SAC, 'TD3':TD3, 'ARS':ARS, 
                        'QRDQN':QRDQN, 'RecurrentPPO':RecurrentPPO, 'TQC':TQC, 'TRPO':TRPO, 'MaskablePPO': MaskablePPO}


    model_class = model_path.split('.')[-1]
    model = model_class_dict[model_class].load(model_path, env=env)

    print ('model:', model, 'path:', model_path)

    vec_env = model.get_env()  
    reward_over_episodes = []
    episode_prices = []
    episode_isin = []

    total_avg_vola_profit = []
    total_prediction_accuracy = []

    total_profit = []
    profit_long = []
    profit_short = []
    loss_long = []
    loss_short = []

    tbar = tqdm(range(total_num_episodes))

    action_sum = 0
    result_counter = {'Action-Sum':0, 'Long-True':0, 'Short-True':0, 'Long-False':0, 'Short-False':0, 'Hold-True':0, 'Hold-False':0}

    obs = vec_env.reset()

    for episode in tbar:

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

        i = info[0]
        episode_isin.append(i['isin'])
        reward_over_episodes.append(i['total_reward'])

        price_tuple = (i['price_open'], i['price_high'], i['price_low'], i['price_close'])
        episode_prices.append(price_tuple)

        total_avg_vola_profit.append(i['total_avg_vola_profit'])
        total_prediction_accuracy.append(i['total_prediction_accuracy'])

        total_profit.append(i['total_profit'])
        profit_long.append(i['total_profit_long'])
        profit_short.append(i['total_profit_short'])
        loss_long.append(i['total_loss_long'])
        loss_short.append(i['total_loss_short'])

    tbar.close()


    print ('Avg. Reward   : ', np.mean(reward_over_episodes))
    print ('median Reward : ', np.median(reward_over_episodes))
    print ('Min  Reward   : ', np.min(reward_over_episodes))
    print ('Max. Reward   : ', np.max(reward_over_episodes))

    result_counter['Action-Sum'] = round(float(action_sum[0]), 3)

    print(result_counter)

    line = '-' * 150
    grey_color = '\033[2m'
    red_color = '\033[31m'
    green_color = '\033[32m'
    normal_color = '\033[0m'
    I = f'{grey_color}|{normal_color}'
    print(f'{grey_color}{line}')
    print(f'{"ISIN":^12} | {"REWARD":^10} | {"PRICE (O / H / L / C)":^35} | {"PROFIT":^12} | {"PROFIT Long":^18} '\
        f'| {"LOSS Long":^18} | {"PROFIT / REWARD":^12}')
    print(f'{"":^12} | {"":^10} | {"":^35} | {"PRED. ACC.":^12} | {"PROFIT Short":^18} | {"LOSS Short":^18} | {"AVG. VOLA PROFIT":^12}')
    print(f'{line}{normal_color}')


    for i in range(len(episode_isin)):
        
        isin = episode_isin[i]
        reward = reward_over_episodes[i]

        _total_profit = total_profit[i]
        _profit_long = profit_long[i]
        _profit_short = profit_short[i]
        _loss_long = loss_long[i]
        _loss_short = loss_short[i]
        _total_avg_vola_profit = total_avg_vola_profit[i]
        _total_prediction_accuracy = total_prediction_accuracy[i]

        if output_min_profit is not None and _total_profit < output_min_profit: continue

        profit_LS = _profit_long + _profit_short
        loss_LS = _loss_long + _loss_short
        total = profit_LS + loss_LS

        profit_long_p = _profit_long / total * 100
        profit_short_p = _profit_short / total * 100
        loss_long_p = _loss_long / total * 100
        loss_short_p = _loss_short / total * 100

        pr = _total_profit / reward * 100 # profit / reward ratio
        price = episode_prices[i]
        O = price[0]
        H = price[1]
        L = price[2]
        C = price[3]
        str_price = f'{O:>6.2f} {H:>6.2f} {L:>6.2f} {C:>6.2f}'

        str_profit = ''
        if _total_profit > 0 : str_profit = f'{green_color}{_total_profit:>12.3f}{normal_color}'
        elif _total_profit < 0 : str_profit = f'{red_color}{_total_profit:>12.3f}{normal_color}'

        print (f'{isin} {I} {reward:10.3f} {I} {str_price:^35} {I} {str_profit} '\
            f'{I} {_profit_long:>9.3f} ({profit_long_p:>5.1f}%) '\
            f'{I} {_loss_long:>9.3f} ({loss_long_p:>5.1f}%) '\
            f'{I} {pr:>10.1f} %')
        
        print (f'{"":^12} {I} {"":^10} {I} {"":^35} {I} {_total_prediction_accuracy:>12.3f} {I} {_profit_short:>9.3f} ({profit_short_p:>5.1f}%) '\
            f'{I} {_loss_short:>9.3f} ({loss_short_p:>5.1f}%) {I} {_total_avg_vola_profit:>10.1f} %')

        print(f'{grey_color}{line}{normal_color}')


#========================================================================================
# configuration
#========================================================================================

#/opt/homebrew/lib/python3.11/site-packages/stable_baselines3/common/save_util.py:166: 
# UserWarning: Could not deserialize object lr_schedule. Consider using `custom_objects` argument to replace this object.
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

window_size = 64 #30
isin_list = [] #if empty -> load from '/Users/alex/Develop/gettex/finanzen.net.pickle'
#isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]
isin_list += ["GB00BYQ0JC66"]

date = None
#date = '2023-04-13' #'2023-04-14'
date = '2023-04-13+14'
max_data = 1 #10 # or None for all data

#model_path = f'./checkpoint/GettexStocks-v0.3500K.29.pred_1.20230419_035236.PPO'
model_path = f'./checkpoint/obs_64/GettexStocks-v0.obs_64.3500K.40.pred_5.20230422_231937.TRPO'

#vec_normalize_path = './checkpoint/GettexStocks-v0.pred_1.vec_normalize.20230419_041750.pkl'
vec_normalize_path = './checkpoint/obs_64/GettexStocks-v0.obs_64.pred_5.vec_normalize.20230423_005402.pkl'


output_min_profit = None #0.0 # or None for no filter

prediction_offset_list = [5]

for prediction_offset in prediction_offset_list:
    test_model(window_size, prediction_offset, max_data, isin_list, date, output_min_profit, model_path, vec_normalize_path)

