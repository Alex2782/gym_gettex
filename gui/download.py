import requests
import json
import pandas as pd
import plotly.express as px
import datetime
import time
import os


headers = {"Accept-Encoding": "gzip",
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
           }

# -------------------------------------------------
# get_content
# -------------------------------------------------
def get_content(url):
    
    response = requests.get(url, headers=headers).json()
    return response

# -------------------------------------------------
# get_jsonfile
# -------------------------------------------------
def get_jsonfile(path):

    with open(path, 'r') as file:
        json_data = file.read()

    return json.loads(json_data)

# ==============================================================

if __name__ == '__main__':


    #TODO Pretrade: https://api.nasdaq.com/api/quote/TSLA/extended-trading?markettype=pre&assetclass=stocks&time=1
    # "filterList":[
    # {"name":"Last 100 Trades","value":0},
    # {"name":"4:00 - 4:29","value":1},
    # {"name":"4:30 - 4:59","value":2},
    # {"name":"5:00 - 5:29","value":3},
    # {"name":"5:30 - 5:59","value":4},
    # {"name":"6:00 - 6:29","value":5},
    # {"name":"6:30 - 6:59","value":6},
    # {"name":"7:00 - 7:29","value":7},
    # {"name":"7:30 - 7:59","value":8},
    # {"name":"8:00 - 8:29","value":9},
    # {"name":"8:30 - 8:59","value":10},
    # {"name":"9:00 - 9:29","value":11}]}

    #TODO Posttrade: https://api.nasdaq.com/api/quote/TSLA/extended-trading?markettype=post&assetclass=stocks&time=1
    # "filterList":[
    # {"name":"Last 100 Trades","value":0},
    # {"name":"4:00 - 4:29","value":1},
    # {"name":"4:30 - 4:59","value":2},
    # {"name":"5:00 - 5:29","value":3},
    # {"name":"5:30 - 5:59","value":4},
    # {"name":"6:00 - 6:29","value":5},
    # {"name":"6:30 - 6:59","value":6},
    # {"name":"7:00 - 7:29","value":7},
    # {"name":"7:30 - 7:59","value":8}]}



    time_now = datetime.datetime.now().time()
    #TODO check US-Time 16:02 - 21:00

    target_time_min = datetime.time(3, 0) # 03:00
    target_time_max = datetime.time(22, 2)  # 22:02

    if time_now > target_time_min and time_now < target_time_max:
        print ('No download required.')
        exit()

    symbol = 'TSLA'
    limit = 10000

    start_time = datetime.time(9, 30)
    end_time = datetime.time(15, 30)
    interval = datetime.timedelta(minutes=30)

    current_time = datetime.datetime.combine(datetime.date.today(), start_time)
    date_str = current_time.strftime("%Y-%m-%d")
    
    _DF = pd.DataFrame()

    while current_time.time() <= end_time:

        time_str = current_time.strftime("%H:%M")
        current_time += interval

        offset = 0
        totalRecords = float('inf')  # Initialisieren mit einem hohen Wert


        _DF_local = pd.DataFrame()

        while offset < totalRecords:
            url = f'https://api.nasdaq.com/api/quote/TSLA/realtime-trades?&limit={limit}&offset={offset}&fromTime={time_str}'
            data = get_content(url) 
            
            data = data['data']
            totalRecords = data['totalRecords']

            if totalRecords == 0: 
                print ('CONTINUE: no data')
                continue

            offset += limit

            rows = data['rows']
            print (f'{time_str} | {offset:>9} / {totalRecords}, {url}')

            df = pd.DataFrame(rows)
            df["nlsTime"] = pd.to_datetime(f'{date_str} ' + df["nlsTime"])
            df["nlsPrice"] = df["nlsPrice"].str.replace("$", "", regex=False).astype(float)
            df["nlsShareVolume"] = df["nlsShareVolume"].str.replace(",", "").astype(int)

            _DF_local = pd.concat([_DF_local, df])
            time.sleep(2)
        
        reversed_df = _DF_local[::-1].reset_index(drop=True)
        _DF = pd.concat([_DF, reversed_df])  

    print (_DF)

    #TODO config for 'output_path'
    output_path = f'/Volumes/GETTEX/nasdaq-data/{symbol}.{date_str}.feather'
    
    if os.path.exists(output_path):
        for i in range(10):
            output_path = f'/Volumes/GETTEX/nasdaq-data/{symbol}.{date_str}.idx_{i}.feather'
            if not os.path.exists(output_path): break

    if len(_DF) > 0:
        _DF.reset_index(inplace=True, drop=True)
        _DF.to_feather(output_path)

    _DF['nlsTime_minutes'] = _DF['nlsTime'].dt.strftime('%Y-%m-%d %H:%M')
    df_grp = _DF.groupby('nlsTime_minutes').agg({'nlsPrice': 'mean', 'nlsShareVolume': 'sum'}).reset_index()
    fig = px.bar(df_grp, x='nlsTime_minutes', y='nlsShareVolume', hover_data=['nlsPrice'])
    fig.show()