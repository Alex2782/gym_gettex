import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import pickle
import math

class GettexStocksEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, frame_bound, prediction_offset=1, render_mode=None):
        assert len(frame_bound) == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.frame_bound = frame_bound
        self.render_mode = render_mode
        self.prediction_offset = prediction_offset

        if type(df) is not list: df = [dict(df=df, frame_bound=self.frame_bound)] 

        self.df = df
        self.df_len = len(self.df)
        self.df_current_idx = 0

        self.window_size = window_size
        #self.shape = (window_size, )#(window_size * 2, )
        # Date,HH,MM,Open,High,Low,Close,Volume,Volume_Ask,Volume_Bid,no_pre_bid,no_pre_ask,
        # no_post,vola_profit,bid_long,bid_short,ask_long,ask_short
        self.shape = (window_size * 19, ) # 18 cols + 1 col for 'diff'

        self.max_volatility = 10.0

        # spaces
        self.action_space = spaces.Box(
            low=-self.max_volatility, high=self.max_volatility, shape=(1,), dtype=np.float32
        )
        INF = 1e10
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = None
        self._terminated = None
        self._current_tick = None
        self._total_reward = None
        self._total_profit = None

        self._prepare_data()
        

    def get_data_idx(self):
        return self._random_list[self.df_current_idx]

    def _prepare_data(self):

        for i in range(self.df_len):
            self.prices, self.signal_features = self._process_data(i)
            self.df[i]['prices'] = self.prices
            self.df[i]['signal_features'] = self.signal_features

    def _create_observation_cache(self):
        """ create cache for faster training """
        self._observation_cache = []

        for current_tick in range(self._start_tick, self._end_tick + 1):
            obs = self.signal_features[(current_tick-self.window_size+1):current_tick+1].flatten()
            self._observation_cache.append(obs)

        pass


    def _process_data(self, df_idx):

        df_dict = self.df[df_idx]
        df = df_dict['df']
        self.frame_bound = df_dict['frame_bound']

        # Date,HH,MM,Open,High,Low,Close,Volume,Volume_Ask,Volume_Bid,no_pre_bid,no_pre_ask,
        # no_post,vola_profit,bid_long,bid_short,ask_long,ask_short
        date = df.loc[:, 'Date'].to_numpy() - datetime.now().year * 10000 # subtract year from date
        HH = df.loc[:, 'HH'].to_numpy()
        MM = df.loc[:, 'MM'].to_numpy()

        open = df.loc[:, 'Open'].to_numpy()
        high = df.loc[:, 'High'].to_numpy()
        low = df.loc[:, 'Low'].to_numpy()
        volume = df.loc[:, 'Volume'].to_numpy()
        volume_ask = df.loc[:, 'Volume_Ask'].to_numpy()
        volume_bid = df.loc[:, 'Volume_Bid'].to_numpy()

        prices = df.loc[:, 'Close'].to_numpy()

        no_pre_bid = df.loc[:, 'no_pre_bid'].to_numpy()
        no_pre_ask = df.loc[:, 'no_pre_ask'].to_numpy()
        no_post = df.loc[:, 'no_post'].to_numpy()
        vola_profit = df.loc[:, 'vola_profit'].to_numpy()
        bid_long = df.loc[:, 'bid_long'].to_numpy()
        bid_short = df.loc[:, 'bid_short'].to_numpy()
        ask_long = df.loc[:, 'ask_long'].to_numpy()
        ask_short = df.loc[:, 'ask_short'].to_numpy()


        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        open[self.frame_bound[0] - self.window_size]
        open = open[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        high[self.frame_bound[0] - self.window_size]
        high = high[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        low[self.frame_bound[0] - self.window_size]
        low = low[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        volume[self.frame_bound[0] - self.window_size]
        volume = volume[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        #signal_features = diff #np.column_stack((prices, diff))
        data_tuple = (date, HH, MM, prices, diff, open, high, low, volume, volume_ask, volume_bid, \
                      no_pre_bid, no_pre_ask, no_post, vola_profit, bid_long, bid_short, ask_long, ask_short)
        
        signal_features = np.column_stack(data_tuple)

        #free memory
        del df_dict['df']
        del df

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0

        current_price = self.prices[self._current_tick]
        if current_price == 0 or math.isnan(current_price): current_price = 0.001
        next_price = self.prices[self._current_tick + self.prediction_offset]
        price_diff = (next_price - current_price) / current_price * 100

        predict_diff = price_diff - action[0]
        if predict_diff > self.max_volatility: predict_diff = self.max_volatility
        elif predict_diff < -self.max_volatility: predict_diff = -self.max_volatility

        #false prediction = -0.85 reward
        if (price_diff < 0 and action[0] > 0) or (price_diff > 0 and action[0] < 0): step_reward = -0.85
        else: 
            step_reward = 1.0 - abs(predict_diff) / self.max_volatility
            self._total_profit += abs(price_diff)

            if price_diff > 0: self._total_profit_long += price_diff
            else: self._total_profit_short += abs(price_diff)

        #print (f'step_reward: {step_reward:5.3f}, action: {action[0]:5.3f}, '\
        #       f'current_price: {current_price:5.3f}, next_price: {next_price:5.3f} = {price_diff:.3f} %, '\
        #       f'predict_diff:{predict_diff:.3f}, total_profit:{self._total_profit:.3f}')

        return step_reward
    
    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            total_profit_long = self._total_profit_long,
            total_profit_short = self._total_profit_short
        )


    def _get_observation(self):
        return self._observation_cache[self._current_tick-self.window_size]

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #print(f'RESET:  {self.df_current_idx}')

        if self.df_current_idx == 0:
            self._random_list = list(range(self.df_len))
            np.random.shuffle(self._random_list)

        self.df_current_idx = (self.df_current_idx + 1) % self.df_len

        idx = self.get_data_idx()
        self.prices = self.df[idx]['prices']
        self.signal_features  = self.df[idx]['signal_features']       
        self._end_tick = len(self.prices) - 1

        self._create_observation_cache()

        self._terminated = False
        self._current_tick = self._start_tick
        self._total_reward = 0.
        self._total_profit = 0.
        self._total_profit_long = 0.
        self._total_profit_short = 0.
        self._first_rendering = True

        info = self._get_info()
        observation = self._get_observation()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info


    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        if self._current_tick + self.prediction_offset  == self._end_tick:
            self._terminated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        observation = self._get_observation()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, step_reward, self._terminated, False, info



# ============================================
# DEBUG Env.
# ============================================

if __name__ == '__main__':

    # -------------------------------------------------
    # load_dict_data
    # -------------------------------------------------
    def load_dict_data(pickle_path = '../finanzen.net.pickle'):
        
        ret = None
        with open(pickle_path, 'rb') as f_in:
            ret = pickle.loads(f_in.read())

        return ret

    def debug_env():
        import sys
        sys.path.append('./') # optional (if not installed via 'pip' -> ModuleNotFoundError)
        import gym_gettex

        import pandas as pd

        #isin_list = []
        #isin_list += ["DE0007236101", "DE0008232125", "US83406F1021", "FI0009000681"]
        #isin_list += ["US4581401001", "NL0011821202", "DE0005552004", "US02079K3059", "US5949181045"]
        #isin_list += ["US88160R1014", "DE000BASF111", "DE000BAY0017", "DE000BFB0019"]
        #isin_list += ["DE0005008007", "DE0005009740", "DE0005019004", "DE0005019038", 
        #            "DE0005032007", "DE0005089031", "DE0005093108", "DE0005102008",
        #            "DE0005103006", "DE0005104400", "DE0005104806", "DE0005110001"]

        # https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
        # create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
        pickle_path = f'/Users/alex/Develop/gettex/finanzen.net.pickle'
        isin_list = load_dict_data(pickle_path)['AKTIE']['isin_list']

        date = None
        #date = '2023-03-29'

        window_size = 30 #15
        prediction_offset = 4 #1

        np.random.shuffle(isin_list)

        _MAX_DATA = None #1500
        if _MAX_DATA is not None and len(isin_list) > _MAX_DATA: isin_list = isin_list[:_MAX_DATA-1]


        df_list = []
        for isin in isin_list:
            
            filename = f'{isin}.csv'
            if date is not None: filename = f'{isin}.{date}.csv'
            path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

            if not os.path.exists(path): 
                print (f'not exists: {path}')
                continue

            df = pd.read_csv(path)
            #print ('path:', path, df)
            start_index = window_size
            #end_index = start_index + 50
            end_index = len(df)
            #end_index = len(df) - 300
            df_dict = dict(df=df, frame_bound = (start_index, end_index))
            df_list.append(df_dict)


        total_num_episodes = len(isin_list) #* 3
        del isin_list

        #import time
        #time.sleep(10)
        #exit()

        env = gym.make('GettexStocks-v0',    
            render_mode = None, #"human",
            df = df_list, #df,
            prediction_offset = prediction_offset,
            window_size = window_size,
            frame_bound = (start_index, end_index))
        
        #print (env.signal_features)
        #print (env.action_space)
        #print (env.observation_space)

        
        #obs, info = env.reset()
        #print (obs, info)
        #obs = env.reset()
        #print (obs)

        tbar = tqdm(range(total_num_episodes))

        reward_over_episodes = []
        action_sum = 0

        for episode in tbar:
            step = 1
            obs = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                action_sum += action[0]
                #print ('-'*10, step, '-'*10)
                obs, reward, terminated, truncated, info = env.step(action)
                #print (reward, terminated, truncated, info)
                done = terminated or truncated
                step += 1

            reward_over_episodes.append(info['total_reward'])
            
        print (f'Avg. Reward   : {np.mean(reward_over_episodes):.3f}' )
        print (f'median Reward : {np.median(reward_over_episodes):.3f}')
        print (f'Min  Reward   : {np.min(reward_over_episodes):.3f}')
        print (f'Max. Reward   : {np.max(reward_over_episodes):.3f}')
        print (f'action_sum    : {action_sum:.3f}')
        
    # --------------------------------------------------------------------------------------------------

    debug_env()