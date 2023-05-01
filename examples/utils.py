import pickle
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go

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
        
        path = get_csv_path(isin, date)

        if not os.path.exists(path): 
            #print (f'not exists: {path}')
            skip_counter += 1
            continue

        df = pd.read_csv(path, dtype=float)

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
# show_predict_stats
#-----------------------------
def show_predict_stats(isin, date, predict_stats):

    path = get_csv_path(isin, date)
    df = pd.read_csv(path)
    tm = df['Date'].astype(str) + ' ' + df['HH'].astype(str) + ':' + df['MM'].astype(str)
    df['DateTime'] = pd.to_datetime(tm)

    candlestick_data = go.Candlestick(x=df['DateTime'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name=isin)  

    columns = ['Predict_DateTime', 'Reward', 'Current_Price', 'Next_Price', 'Action', 'Predict_Price']
    predict_df = pd.DataFrame(predict_stats, columns=columns)
    #print (predict_df)

    predict_marker = []
    predict_color = []
    predict_info = []

    for index, row in predict_df.iterrows():
        action = row['Action']
        reward = row['Reward']
        predict_price = row['Predict_Price']
        current_price = row['Current_Price']
        next_price = row['Next_Price']
        price_diff = next_price - current_price

        if action > 0: marker = 'triangle-up'
        else: marker = 'triangle-down'
        
        if reward > 0: color = '#8cd4c8'
        elif reward < 0: color = '#c87ad6'
        else: color = 'silver'

        info = (
            f'predict: {action:.3f}<br>'
            f'reward: {reward:.3f}<br>'
            f'price: {predict_price:.3f}<br>'
            f'old price: {current_price:.3f}<br>'
            f'new price: {next_price:.3f}<br>'
            f'diff: {price_diff:.3f}<br>'
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
                            name='Predict')


    fig = go.Figure(data=[candlestick_data, marker_data])

    # Diagramm layout konfigurieren
    fig.update_layout(title=f'ISIN {isin} prediction',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False)

    fig.show()