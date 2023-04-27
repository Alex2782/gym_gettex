import pickle
import numpy as np
import os
import pandas as pd

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
        
        filename = f'{isin}.csv'
        if date is not None: filename = f'{isin}.{date}.csv'
        path = f'/Users/alex/Develop/gettex/data_ssd/{filename}'

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