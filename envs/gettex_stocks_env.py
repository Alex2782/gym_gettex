import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import math
class GettexStocksEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df_list = None, window_size = 15, prediction_offset=1, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.prediction_offset = prediction_offset
        self.window_size = window_size

        #self.shape = (window_size, )#(window_size * 2, )
        # Date,HH,MM,Open,High,Low,Close,Volume,Volume_Ask,Volume_Bid,no_pre_bid,no_pre_ask,
        # no_post,vola_profit,bid_long,bid_short,ask_long,ask_short
        self.shape = (window_size * 20, ) # 18 cols + 2 col for 'diff' and 'weekdays'

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

        self._isin = None
        self._predict_datetime = None
        self._price_open = None
        self._price_high = None
        self._price_low = None
        self._price_close = None

        self._current_price = None
        self._next_price = None
        self._predict_price = None
        self._avg_vola_profit = None

        if df_list is not None:
            self.init_df_list(df_list)

    def init_df_list(self, df_list):
        self.df_list = df_list
        self.df_len = len(self.df_list)
        self.df_current_idx = 0
        self._prepare_data()

    def get_data_idx(self):
        return self._random_list[self.df_current_idx]

    def _prepare_data(self):

        for i in range(self.df_len):
            isin, self.prices, self.signal_features, self.datetime_list = self._process_data(i)

            self.df_list[i]['isin'] = isin
            self.df_list[i]['prices'] = self.prices
            self.df_list[i]['signal_features'] = self.signal_features
            self.df_list[i]['datetime_list'] = self.datetime_list

    def _create_observation_cache(self):
        """ create cache for faster training """
        self._observation_cache = []

        for current_tick in range(self._start_tick - 1, self._end_tick + 1):
            start = current_tick - self.window_size + 1
            end = current_tick + 1
            obs = self.signal_features[start:end].flatten()
            self._observation_cache.append(obs)

        pass


    def _process_data(self, df_idx):

        df_dict = self.df_list[df_idx]
        df = df_dict['df']
        frame_bound = df_dict['frame_bound']
        isin = df_dict['isin']

        # Date,HH,MM,Open,High,Low,Close,Volume,Volume_Ask,Volume_Bid,no_pre_bid,no_pre_ask,
        # no_post,vola_profit,bid_long,bid_short,ask_long,ask_short
        dt = df['Date'].astype(int).astype(str) + ' ' + df['HH'].astype(int).astype(str) + ':' + df['MM'].astype(int).astype(str)
        datetime_list = pd.to_datetime(dt)        

        date = df.loc[:, 'Date'].to_numpy()
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
    
        start = frame_bound[0]-self.window_size
        end = frame_bound[1]

        date = date[start:end]

        weekdays = []
        for d in date:
            
            date_obj = datetime.strptime(str(int(d)), '%Y%m%d')
            weekday_int = date_obj.weekday()
            weekdays.append(weekday_int)
            #print (d, weekday_int)


        date = date - datetime.now().year * 10000 # subtract year from date

        HH = HH[start:end]
        MM = MM[start:end]

        prices = prices[start:end]

        open = open[start:end]
        high = high[start:end]
        low = low[start:end]
        diff = np.insert(np.diff(prices), 0, 0)

        volume = volume[start:end]
        volume_ask = volume_ask[start:end]
        volume_bid = volume_bid[start:end]
        no_pre_bid = no_pre_bid[start:end]
        no_pre_ask = no_pre_ask[start:end]
        no_post = no_post[start:end]
        vola_profit = vola_profit[start:end]
        bid_long = bid_long[start:end]
        bid_short = bid_short[start:end]
        ask_long = ask_long[start:end]
        ask_short = ask_short[start:end]

        data_tuple = (date, weekdays, HH, MM, prices, diff, open, high, low, volume, volume_ask, volume_bid, \
                      no_pre_bid, no_pre_ask, no_post, vola_profit, bid_long, bid_short, ask_long, ask_short)
        
        signal_features = np.column_stack(data_tuple)

        #free memory
        del df_dict['df']
        del df

        return isin, prices, signal_features, datetime_list

    def get_info(self):
        return dict(
            isin = self._isin,
            predict_datetime = self._predict_datetime,
            total_reward = self._total_reward,

            total_profit = self._total_profit,
            total_profit_long = self._total_profit_long,
            total_profit_short = self._total_profit_short,

            total_loss = self._total_loss,
            total_loss_long = self._total_loss_long,
            total_loss_short = self._total_loss_short,

            total_avg_vola_profit = self._total_avg_vola_profit,
            total_prediction_accuracy = self._total_prediction_accuracy,

            price_open = self._price_open,
            price_high = self._price_high,
            price_low = self._price_low,
            price_close = self._price_close,

            current_price = self._current_price,
            next_price = self._next_price,
            predict_price = self._predict_price,
            avg_vola_profit = self._avg_vola_profit
        )


    def get_observation(self):
        return self._observation_cache[self._current_tick-self.window_size]

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #print(f'RESET:  {self.df_current_idx}')

        if self.df_current_idx == 0:
            self._random_list = list(range(self.df_len))
            np.random.shuffle(self._random_list)

        self.df_current_idx = (self.df_current_idx + 1) % self.df_len

        idx = self.get_data_idx()
        self.prices = self.df_list[idx]['prices']
        self._isin = self.df_list[idx]['isin']

        self._price_open = self.prices[0]
        self._price_high = np.max(self.prices)
        self._price_low = np.min(self.prices)
        self._price_close = self.prices[-1]

        self.signal_features  = self.df_list[idx]['signal_features']      
        self.datetime_list  = self.df_list[idx]['datetime_list']
        self._end_tick = len(self.prices) - 1

        #print ('signal_features:', self.signal_features)

        self._create_observation_cache()

        #print ('len signal_features   :', len(self.signal_features))
        #print ('len _observation_cache:', len(self._observation_cache))


        self._terminated = False
        self._current_tick = self._start_tick
        self._total_reward = 0.

        self._predict_datetime = None
        predict_idx = self._start_tick + self.prediction_offset - 1
        if predict_idx < self._end_tick:
            self._predict_datetime = self.datetime_list[predict_idx]
        
        self._total_profit = 0.
        self._total_profit_long = 0.
        self._total_profit_short = 0.
        
        self._total_loss = 0.
        self._total_loss_long = 0.
        self._total_loss_short = 0.
        
        self._total_avg_vola_profit = 0.
        self._total_prediction_accuracy = 0.

        self._current_price = 0.
        self._next_price = 0.
        self._predict_price = 0.
        self._avg_vola_profit = 0.

        self._first_rendering = True

        observation = self.get_observation()
        info = self.get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info


    def _calculate_reward(self, action):
        step_reward = 0
        action = action[0]

        start = self._current_tick - 1
        end = self._current_tick + self.prediction_offset - 1

        self._predict_datetime = self.datetime_list[end]
        predict_signals = self.signal_features[start:end+1]
        
        #data_tuple = (date, weekdays, HH, MM, prices, diff, open, high, low, volume, volume_ask, volume_bid, \
        #              no_pre_bid, no_pre_ask, no_post, vola_profit, bid_long, bid_short, ask_long, ask_short)
        last_signal = predict_signals[-1]
        high = last_signal[7]
        low = last_signal[8]
        no_pre_bid = last_signal[12]
        no_pre_ask = last_signal[13]
        vola_profit = last_signal[15]
        bid_long = last_signal[16]
        bid_short = last_signal[17]
        ask_long = last_signal[18]
        ask_short = last_signal[19]

        #workaround: no bid and ask, high vola_profit, very high difference bid <-> ask
        TRADE_ERROR = False 
        long_diff = abs(ask_long - bid_long)
        short_diff = abs(ask_short - bid_short)
        trade_diff = abs(short_diff - long_diff)
        if no_pre_bid > 0 and no_pre_ask > 0 and vola_profit > 3 and trade_diff > vola_profit:
            TRADE_ERROR = True
            #print ('last_signal:', no_pre_bid, no_pre_ask, vola_profit)
            #print ('TRADE_ERROR:', TRADE_ERROR, 'trade_diff:', trade_diff, long_diff, short_diff)
            #for signal in predict_signals:
            #    print ('signal:', signal[0], signal[1], signal[2], signal[3], signal[4], signal[5])
            #print('-'*10)

        current_price = self.prices[start]
        next_price = self.prices[end]
        predict_price = 0
        avg_vola_profit = 0

        if not TRADE_ERROR:

            vola_profit_list = []
            #print ('_calculate_reward:', start, end)
            for signal in predict_signals:
                vola_profit_list.append(signal[15])
            #    print ('signal:', signal[1], signal[2], signal[3])
            #print('-'*10)
            
            avg_vola_profit = np.mean(vola_profit_list)

            self._total_avg_vola_profit = (self._total_avg_vola_profit + avg_vola_profit) / 2

            
            if current_price == 0 or math.isnan(current_price): current_price = 0.001
            price_diff = (next_price - current_price) / current_price * 100

            #print (current_price, next_price, price_diff)

            predict_diff = price_diff - action
            if predict_diff > self.max_volatility: predict_diff = self.max_volatility
            elif predict_diff < -self.max_volatility: predict_diff = -self.max_volatility

            predict_price = current_price + (current_price * action / 100)

            self._total_prediction_accuracy = (self._total_prediction_accuracy + abs(predict_diff)) / 2

            tolerance = (low + (high - low) / 2) * 0.0001

            if predict_price >= low - tolerance and predict_price <= high + tolerance:
                step_reward = abs(action)
                self._total_profit += abs(step_reward)
                if action > 0: self._total_profit_long += action
                else: self._total_profit_short += abs(action)                
            else:
                step_reward = -abs(action)
                self._total_profit -= abs(step_reward)
                if action > 0: self._total_loss_long += action
                else: self._total_loss_short += abs(action)

              


        #print (f'step_reward: {step_reward:5.3f}, action: {action:5.3f}, '\
        #       f'current_price: {current_price:5.3f}, next_price: {next_price:5.3f} = {price_diff:.3f} %, '\
        #       f'predict_diff:{predict_diff:.3f}, total_profit:{self._total_profit:.3f}')

        self._current_price = current_price
        self._next_price = next_price
        self._predict_price = predict_price
        self._avg_vola_profit = avg_vola_profit

        return step_reward

    def step(self, action):
        #print ('STEP, _current_tick: ', self._current_tick)

        self._terminated = False
        self._current_tick += 1

        #print (self._current_tick + self.prediction_offset - 1, '==', self._end_tick)
        if self._current_tick + self.prediction_offset - 1  == self._end_tick:
            self._terminated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        observation = self.get_observation()
        info = self.get_info()

        return observation, step_reward, self._terminated, False, info

    def close(self):
        del self._observation_cache
        del self.signal_features
        del self.datetime_list
        del self.df_list



# ============================================
# DEBUG Env.
# ============================================
if __name__ == '__main__':
        
    import sys
    sys.path.append('./') # optional (if not installed via 'pip' -> ModuleNotFoundError)
    import gym_gettex
    from gym_gettex.examples.utils import load_data, get_finanzen_stock_isin_list, show_predict_stats

    #-----------------------------
    # debug_env
    #-----------------------------
    def debug_env():

        window_size = 64 #30 #15
        prediction_offset = 4 #1

        max_data = 1 #256 #None #10
        isin_list = []
        isin_list += ["GB00BYQ0JC66"]
        date = None
        date = '2023-04-13+14'

        #isin_list, df_list = load_data(window_size, isin_list, date, max_data)

        env = gym.make('GettexStocks-v0',    
            render_mode = None, #"human",
            #df = df_list, #optional
            prediction_offset = prediction_offset,
            window_size = window_size)
        
        #print (env.signal_features)
        #print (env.action_space)
        #print (env.observation_space)

        #obs, info = env.reset()
        #print (obs, info)
        #obs = env.reset()
        #print (obs)


        #isin_list = get_finanzen_stock_isin_list()
        #np.random.shuffle(isin_list)

        batch_list = np.array_split(isin_list, len(isin_list)/max_data)
        len_batch = len(batch_list)

        for i in range (0, len_batch):

            print ('BATCH:', i + 1, ' / ', len_batch)
            print ('-' * 50)

            batch_isin = batch_list[i]

            batch_isin, df_list, skip_counter = load_data(window_size, batch_isin, date, None)
            total_num_episodes = len(df_list)
            
            env.init_df_list(df_list)

            tbar = tqdm(range(total_num_episodes))

            reward_over_episodes = []
            action_sum = 0

            predict_stats = []

            for episode in tbar:
                step = 1

                obs, info = env.reset()
                done = False
                isin = info['isin']

                while not done:
                    action = env.action_space.sample()
                    #action = [-0.1]
                    action_sum += action[0]
                    obs, reward, terminated, truncated, info = env.step(action)

                    predict_datetime = info['predict_datetime']
                    current_price = info['current_price']
                    next_price = info['next_price']
                    predict_price = info['predict_price']

                    predict_stats.append([predict_datetime, reward, current_price, next_price, action[0], predict_price])

                    #print (obs)
                    #print (reward, terminated, truncated, info)
                    done = terminated or truncated
                    step += 1

                reward_over_episodes.append(info['total_reward'])
                
            print(f'Avg. Reward   : {np.mean(reward_over_episodes):.3f}' )
            print(f'median Reward : {np.median(reward_over_episodes):.3f}')
            print(f'Min  Reward   : {np.min(reward_over_episodes):.3f}')
            print(f'Max. Reward   : {np.max(reward_over_episodes):.3f}')
            print(f'action_sum    : {action_sum:.3f}')
            print(info)

            show_predict_stats = True
            if show_predict_stats:
                show_predict_stats(isin, date, predict_stats)

            del df_list
        
        env.close()
    # --------------------------------------------------------------------------------------------------

    debug_env()