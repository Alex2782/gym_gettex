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
# output_over_batch_stats
# -------------------------------------------------------------------------------------
def output_over_batch_stats(reward_over_batch, total_profit_over_batch, total_avg_vola_profit_over_batch):
    print ('='*20, 'OVER BATCH STATS', '='*20)

    print ('Avg. Reward   : ', np.mean(reward_over_batch))
    print ('median Reward : ', np.median(reward_over_batch))
    print ('Min  Reward   : ', np.min(reward_over_batch))
    print ('Max. Reward   : ', np.max(reward_over_batch))
    print ('-'*60)
    print ('Avg. Profit   : ', np.mean(total_profit_over_batch))
    print ('median Profit : ', np.median(total_profit_over_batch))
    print ('Min  Profit   : ', np.min(total_profit_over_batch))
    print ('Max. Profit   : ', np.max(total_profit_over_batch))
    print ('-'*60)
    print ('Avg. Vol. Profit   : ', np.mean(total_avg_vola_profit_over_batch))
    print ('median Vol. Profit : ', np.median(total_avg_vola_profit_over_batch))
    print ('Min  Vol. Profit   : ', np.min(total_avg_vola_profit_over_batch))
    print ('Max. Vol. Profit   : ', np.max(total_avg_vola_profit_over_batch))

# -------------------------------------------------------------------------------------
# output_stats
# -------------------------------------------------------------------------------------
def output_stats(reward_over_episodes, result_counter, action_sum, episode_isin, total_profit, profit_long,
                 profit_short, loss_long, loss_short, total_avg_vola_profit, total_prediction_accuracy, episode_prices):

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


# -------------------------------------------------------------------------------------
# test_model
# -------------------------------------------------------------------------------------
def test_model(window_size=30, prediction_offset=2, max_data=256, isin_list = [], date = None, 
               output_min_profit=0.0, model_path=None, vec_normalize_path=None, _show_predict_stats=False):

    env_name = 'GettexStocks-v0'

    env = gym.make(env_name,    
        render_mode = None, #"human",
        #df = df_list, #optional
        prediction_offset = prediction_offset,
        window_size = window_size
    )

    orig_env = env

    env = DummyVecEnv([lambda: env])

    if vec_normalize_path is not None:
        env = VecNormalize.load(vec_normalize_path, env)

    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    model_class_dict = {'A2C':A2C, 'DDPG':DDPG, 'DQN':DQN, 'PPO':PPO, 'SAC':SAC, 'TD3':TD3, 'ARS':ARS, 
                        'QRDQN':QRDQN, 'RecurrentPPO':RecurrentPPO, 'TQC':TQC, 'TRPO':TRPO, 'MaskablePPO': MaskablePPO}

    model_class = model_path.split('.')[-1]
    model = model_class_dict[model_class].load(model_path, env=env)

    print ('model:', model, 'path:', model_path)
    vec_env = model.get_env()  


    # https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
    # create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
    if isin_list is None or len(isin_list) == 0:
        isin_list = get_finanzen_stock_isin_list()
        np.random.shuffle(isin_list)

    batch_list = np.array_split(isin_list, len(isin_list)/max_data + 1)
    len_batch = len(batch_list)

    min_df_len = window_size + prediction_offset + 1

    reward_over_batch = []
    total_profit_over_batch = []
    total_avg_vola_profit_over_batch = []

    for i in range (0, len_batch):

        print ('BATCH:', i + 1, ' / ', len_batch)
        print ('-' * 50)

        batch_isin = batch_list[i]

        batch_isin, df_list, skip_counter = load_data(window_size, batch_isin, date, None, min_df_len)
        print (f'{skip_counter} files skipped')
        
        orig_env.init_df_list(df_list)

        action_sum = 0

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

        total_num_episodes = len(df_list) - skip_counter
        tbar = tqdm(range(total_num_episodes))

        action_sum = 0
        result_counter = {'Action-Sum':0, 'Long-True':0, 'Short-True':0, 'Long-False':0, 'Short-False':0, 'Hold-True':0, 'Hold-False':0}

        obs = vec_env.reset()

        info = orig_env.get_info()
        isin = info['isin']
        predict_stats = []

        for episode in tbar:

            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)

                if _show_predict_stats:
                    predict_datetime = info[0]['predict_datetime']
                    current_price = info[0]['current_price']
                    next_price = info[0]['next_price']
                    predict_price = info[0]['predict_price']
                    _total_profit = info[0]['total_profit']
                    avg_vola_profit = info[0]['avg_vola_profit']

                    predict_stats.append([predict_datetime, reward[0], current_price, next_price, 
                                          action[0][0], predict_price, _total_profit, avg_vola_profit])

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

            if _show_predict_stats and len(isin_list) < 6:
                show_predict_stats(isin, date, predict_stats, i['total_profit'], i['total_profit_long'], i['total_profit_short'])

        tbar.close()


        output_stats(reward_over_episodes, result_counter, action_sum, episode_isin, total_profit, profit_long,
                 profit_short, loss_long, loss_short, total_avg_vola_profit, total_prediction_accuracy, episode_prices)
        
        reward_over_batch += reward_over_episodes 
        total_profit_over_batch += total_profit
        total_avg_vola_profit_over_batch += total_avg_vola_profit
    
    #output 'OVER BATCH STATS'
    output_over_batch_stats(reward_over_batch, total_profit_over_batch, total_avg_vola_profit_over_batch)



#========================================================================================
# configuration
#========================================================================================

#/opt/homebrew/lib/python3.11/site-packages/stable_baselines3/common/save_util.py:166: 
# UserWarning: Could not deserialize object lr_schedule. Consider using `custom_objects` argument to replace this object.
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

window_size = 64
isin_list = [] #if empty -> load from '/Users/alex/Develop/gettex/finanzen.net.pickle'
#isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]
#isin_list += ["GB00BYQ0JC66"]
isin_list += ["US88160R1014"] #tesla
#isin_list += ["DE0008232125"]


date = None
#date = '2023-04-13'
#date = '2023-04-13+14'
max_data = 256 #10 # or None for all data


#model_path = f'./checkpoint/GettexStocks-v0.obs_30.1000K.2.pred_3.20230502_211413.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.1000K.2.pred_3.20230502_210832.PPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.50000K.122.pred_1.20230503_063708.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.50000K.128.pred_2.20230503_162111.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.50000K.113.pred_3.20230503_203847.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.100000K.162.pred_2.20230506_092418.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.150000K.181.pred_3.20230507_113818.TRPO'
#model_path = f'./checkpoint/GettexStocks-v0.obs_30.150000K.158.pred_4.20230507_235919.TRPO'
model_path = f'./checkpoint/GettexStocks-v0.obs_64.150000K.172.pred_1.20230509_135725.TRPO'


#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_3.vec_normalize.20230502_211744.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_1.vec_normalize.20230503_063708.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_2.vec_normalize.20230503_162111.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_3.vec_normalize.20230503_203847.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_2.vec_normalize.20230506_092418.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_3.vec_normalize.20230507_113818.pkl'
#vec_normalize_path = './checkpoint/GettexStocks-v0.obs_30.pred_4.vec_normalize.20230507_235919.pkl'
vec_normalize_path = './checkpoint/GettexStocks-v0.obs_64.pred_1.vec_normalize.20230509_135725.pkl'


output_min_profit = None # or None for no filter
_show_predict_stats = True

prediction_offset_list = [1]

for prediction_offset in prediction_offset_list:
    test_model(window_size, prediction_offset, max_data, isin_list, date, output_min_profit, model_path, vec_normalize_path,
               _show_predict_stats)

