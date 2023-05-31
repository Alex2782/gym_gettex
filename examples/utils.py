import pickle
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pytz import timezone

# -------------------------------------------------
# load_dict_data
# -------------------------------------------------
def load_dict_data(pickle_path = '../finanzen.net.pickle'):
    
    ret = None
    with open(pickle_path, 'rb') as f_in:
        ret = pickle.loads(f_in.read())

    return ret

#-----------------------------
# get_finanzen_stock_isin_list
#-----------------------------    
def get_finanzen_stock_isin_list(pickle_path = '/Users/alex/Develop/gettex/finanzen.net.pickle'):
        return load_dict_data(pickle_path)['AKTIE']['isin_list']    


#-----------------------------
# get_csv_path
#-----------------------------
def get_csv_path(isin, date):

    filename = f'{isin}.csv'
    if date is not None: filename = f'{isin}.{date}.csv'
    path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

    return path

#-----------------------------
# get_feather_path
#-----------------------------
def get_feather_path(isin, date):

    filename = f'{isin}.feather'
    if date is not None: filename = f'{isin}.{date}.feather'
    path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

    return path

#-----------------------------
# load_data
#-----------------------------
def load_data(window_size, isin_list=[], date=None, max_data=100, min_df_len=None):

    # https://mein.finanzen-zero.net/assets/searchdata/downloadable-instruments.csv
    # create pickle file: https://github.com/Alex2782/gettex-import/blob/main/finanzen_net.py
    if isin_list is None or len(isin_list) == 0:
        isin_list = get_finanzen_stock_isin_list()
        np.random.shuffle(isin_list)

    if max_data is not None and len(isin_list) > max_data: isin_list = isin_list[:max_data]

    df_list = []
    skip_counter = 0

    for isin in isin_list:
        
        #path = get_csv_path(isin, date)
        path = get_feather_path(isin, date)

        if not os.path.exists(path): 
            #print (f'not exists: {path}')
            skip_counter += 1
            continue

        #df = pd.read_csv(path, dtype=float)
        df = pd.read_feather(path)

        if min_df_len is not None and min_df_len > len(df):
            skip_counter += 1
            continue             

        start_index = window_size
        end_index = len(df)
        df_dict = dict(isin=isin, df=df, frame_bound=(start_index, end_index))
        df_list.append(df_dict)

    #print ('df.dtype:', df.dtypes)
    #print ('df.memory_usage:', df.memory_usage(deep=True))

    return isin_list, df_list, skip_counter


#-----------------------------
# show_trade_chart
#-----------------------------
def show_trade_chart(isin, chart_out):

    #[row['datetime'], _prev_day_close, _open, _high, _low, _close]
    
    columns = ['datetime', 'prev_day_close', 'open', 'high', 'low', 'close']
    df = pd.DataFrame(chart_out, columns=columns)

    candlestick_data = go.Candlestick(x=df['datetime'],
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name=isin)   

    marker_data = go.Scatter(x=df['datetime'],
                            y=df['prev_day_close'],
                            mode='lines+markers',
                            marker_size=14,
                            opacity=0.9,
                            line=dict(color='#7777ff', width=2),
                            name= 'prev_day_close')


    fig = go.Figure(data=[candlestick_data, marker_data])    

    fig.update_layout(title=isin,
                    yaxis_title='Price',
                    #xaxis_range=['2023-05-01', '2023-05-10'], 
                    #yaxis_range=[140, 160],
                    xaxis_rangeslider_visible=False)

    fig.show()

    pass

#-----------------------------
# show_predict_stats
#-----------------------------
def show_predict_stats(isin, date, predict_stats, total_profit, total_profit_long, total_profit_short):

    path = get_csv_path(isin, date)
    df = pd.read_csv(path)
    tm = df['Date'].astype(str) + ' ' + df['HH'].astype(str) + ':' + df['MM'].astype(str)
    df['DateTime'] = pd.to_datetime(tm).dt.tz_localize('UTC')
    df['DateTime_DE'] = df['DateTime'].dt.tz_convert('Europe/Berlin')
    df['DateTime_US'] = df['DateTime'].dt.tz_convert('US/Eastern')

    candlestick_data = go.Candlestick(x=df['DateTime'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name=isin)  

    columns = ['Predict_DateTime', 'Reward', 'Current_Price', 'Next_Price', 'Action', 
               'Predict_Price', 'Total_Profit', 'Avg_Vola_Profit']
    predict_df = pd.DataFrame(predict_stats, columns=columns)
    #print (predict_df)

    predict_df['Predict_DateTime'] = predict_df['Predict_DateTime'].dt.tz_localize('UTC')
    predict_df['DateTime_DE'] = predict_df['Predict_DateTime'].dt.tz_convert('Europe/Berlin')
    predict_df['DateTime_US'] = predict_df['Predict_DateTime'].dt.tz_convert('US/Eastern')

    predict_marker = []
    predict_color = []
    predict_info = []

    at_us_stock_opening = {}

    for index, row in predict_df.iterrows():
        action = row['Action']
        reward = row['Reward']
        predict_price = row['Predict_Price']
        current_price = row['Current_Price']
        next_price = row['Next_Price']
        total_profit = row['Total_Profit']
        avg_vola_profit = row['Avg_Vola_Profit']

        _time_us = row['DateTime_US'].strftime('%H:%M')
        
        if at_us_stock_opening.get(_time_us) is None: at_us_stock_opening[_time_us] = 0
        at_us_stock_opening[_time_us] += reward

        price_diff = next_price - current_price
        price_diff_p = price_diff / current_price * 100

        if action < 0.05 and action > -0.05: marker = 'circle'
        elif action > 0: marker = 'triangle-up'
        else: marker = 'triangle-down'
        
        if reward > 0: color = '#8cd4c8'
        elif reward < 0: color = '#c87ad6'
        else: color = 'silver'

        info = (
            f'predict: {action:.3f} %<br>'
            f'reward: {reward:.3f}<br>'
            f'pred.price: {predict_price:.3f}<br>'
            f'old price: {current_price:.3f}<br>'
            f'new price: {next_price:.3f}<br>'
            f'diff: {price_diff:.3f} ({price_diff_p:.3f} %)<br>'
            f'total profit: {total_profit:.3f}<br>'
            f'avg.vola.profit: {avg_vola_profit:.3f}<br>'
        )

        predict_marker.append(marker)
        predict_color.append(color)
        predict_info.append(info)

    # predict marker
    marker_data = go.Scatter(x=predict_df['Predict_DateTime'],
                            y=predict_df['Predict_Price'],
                            mode='lines+markers',
                            marker_symbol=predict_marker,
                            marker_color=predict_color,
                            marker_size=14,
                            hovertext=predict_info,
                            opacity=0.9,
                            line=dict(color='#7777ff', width=2),
                            name= 'Predict')


    fig = go.Figure(data=[candlestick_data, marker_data])

    title = f'ISIN: {isin}, profit: {total_profit:.2f} %, long: {total_profit_long:.2f} %, short: {total_profit_short:.2f} %,'\
            f' rewards @09:45 = {np.sum(at_us_stock_opening.get("09:45")):>9.3f}'
    
    fig.update_layout(title=title,
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False)

    fig.show()


    for us_time in at_us_stock_opening:

        print (f'{us_time} = {at_us_stock_opening[us_time]:>9.3f}')



#-----------------------------
# calculate_rsi
#-----------------------------
def calculate_rsi(data, window=14):
    close_prices = data['close'].values
    deltas = np.diff(close_prices)

    # Gewinne und Verluste aufteilen
    gains = deltas.copy()
    losses = deltas.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0

    avg_gains = []
    avg_losses = []

    for i in range(window, len(close_prices)):
        avg_gain = np.mean(gains[i-window:i])  # angepasstes Fenster
        avg_loss = np.abs(np.mean(losses[i-window:i]))  # angepasstes Fenster
        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)

    avg_gains = np.array(avg_gains)
    avg_losses = np.array(avg_losses)

    # RSI-Werte berechnen
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Länge der RSI-Werte anpassen
    rsi = np.concatenate((np.full(window, np.nan), rsi))  # mit NaN-Werten auffüllen

    return rsi

#-----------------------------
# prepare_training_data
#-----------------------------
def prepare_training_data(isin, df):

    #TODO EUR -> USD
    # indicators: EMA, RSI, etc.

    drop_cols = ['bid_size_max', 'bid_size_min', 'ask_size_max', 'ask_size_min', 'spread_max', 'spread_min', 'activity', 'volatility_long', 'volatility_short', 'vola_activity_long', 'vola_activity_short', 'vola_activity_equal']
    df.drop(drop_cols, axis=1, inplace=True)

    df.set_index('datetime', inplace=True)
    df_timeframes = []
    frame_mode_list = ['15T', '30T', '1H', '1D']
    
    for frame_mode in frame_mode_list:
        # T = Minutes, H = Hour, D = Day
        df = df.resample(frame_mode).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })

        df_timeframes.append(df)

    for df in df_timeframes:

        df.dropna(inplace=True)
        berlin_tz = timezone('Europe/Berlin')
        newyork_tz = timezone('America/New_York')

        df.reset_index(inplace=True)
        df['berlin_time'] = df['datetime'].dt.tz_convert(berlin_tz)
        df['newyork_time'] = df['datetime'].dt.tz_convert(newyork_tz)

        df['berlin_hour'] = df['berlin_time'].dt.hour.astype('int8')
        df['berlin_minute'] = df['berlin_time'].dt.minute.astype('int8')
        df['newyork_hour'] = df['newyork_time'].dt.hour.astype('int8')
        df['newyork_minute'] = df['newyork_time'].dt.minute.astype('int8')

        df.drop(['berlin_time', 'newyork_time'], axis=1, inplace=True)

        for window in [5, 9, 12, 26, 54]:
            sma = df['close'].rolling(window=window).mean()
            col_name = 'sma_' + str(window) 
            df[col_name] = sma
            #df.insert(len(df.columns), col_name, sma)

            ema = df['close'].ewm(span=window, adjust=False).mean()
            col_name = 'ema_' + str(window) 
            df[col_name] = ema
            #df.insert(len(df.columns), col_name, ema)

            rsi = calculate_rsi(df, window)
            print ('rsi:', rsi)
            col_name = 'rsi_' + str(window) 
            df[col_name] = rsi
            #df.insert(len(df.columns), col_name, rsi)


    for df in df_timeframes:
        print ('-'*100)
        print (df.head(10))
        print (df.tail(10))
    
    df.info()




