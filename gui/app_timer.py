import dash

from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import datetime
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Dash-Anwendung erstellen
app = dash.Dash(__name__)

print ('START:', app)
symbol = 'TSLA'
date_str = '2023-05-16'
path = f'/Volumes/GETTEX/nasdaq-data/{symbol}.{date_str}.feather'
_DF = pd.read_feather(path)
del _DF['index']
_DF = _DF.reset_index(drop=True)
#_DF['nlsTime'] = _DF['nlsTime'].dt.strftime('%Y-%m-%d %H:%M')
_DF = _DF.set_index('nlsTime')
print(_DF.columns)
print(_DF)

# Layout erstellen
app.layout = html.Div(
    children=[
        html.H1("Timer"),
        html.Button("Start", id="start-button", n_clicks=0),
        html.Button("Stop", id="stop-button", n_clicks=0),
        html.Div(id="timer-output"),
        html.Div(id="interval-output"),
        dcc.Graph(id='barplot-price'),
        dcc.Graph(id='barplot-volume'),
        dcc.Interval(id="interval", interval=1000, disabled=True)
    ]
)

# Callback-Funktion für Timer
@app.callback(
    Output("interval-output", "children"),
    dash.dependencies.Output('barplot-price', 'figure'),
    dash.dependencies.Output('barplot-volume', 'figure'),
    Input("interval", "n_intervals")
)
def update_timer(n):
    print ('update_timer', n)
    current = datetime.datetime.now()
    current_time = current.strftime("%H:%M:%S")
    current_date = current.strftime("%Y-%m-%d")
    
    if _DF is None or len (_DF) == 0:
        url = 'https://api.nasdaq.com/api/quote/TSLA/realtime-trades?&limit=100&offset=0&fromTime=00:00'

        data = get_content(url)
        data = data['data']
        totalRecords = data['totalRecords']
        offset = data['offset']
        limit = data['limit']

        rows = data['rows']
        rows.reverse()

        df = pd.DataFrame(rows)
        df["nlsTime"] = pd.to_datetime(f'{current_date} ' + df["nlsTime"])
        df["nlsPrice"] = df["nlsPrice"].str.replace("$", "", regex=False).astype(float)
        df["nlsShareVolume"] = df["nlsShareVolume"].str.replace(",", "").astype(int)

        df_grp = df.groupby('nlsTime').agg({'nlsPrice': 'mean', 'nlsShareVolume': 'sum'}).reset_index()
        df_candlestick = df.groupby(pd.Grouper(key='nlsTime', freq='S')).agg({'nlsPrice': ['min', 'max', 'first', 'last']})
    
    else:
        df = _DF
        df_grp = _DF.groupby(pd.Grouper(freq='Min')).agg({'nlsPrice': 'mean', 'nlsShareVolume': 'sum'}).reset_index()
        #df_candlestick = _DF.groupby(pd.Grouper(key='nlsTime', freq='S')).agg({'nlsPrice': ['min', 'max', 'first', 'last']})
        df_candlestick = _DF.groupby(pd.Grouper(freq='Min')).agg({'nlsPrice': ['min', 'max', 'first', 'last']})
    

    fig_volume = px.bar(df_grp, x='nlsTime', y='nlsShareVolume', hover_data=['nlsPrice'])

    fig_volume.update_layout(
        title='Volume',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Volume')
    )

    candlestick_data = go.Candlestick(
        x=df_candlestick.index,
        open=df_candlestick[('nlsPrice', 'first')],
        high=df_candlestick[('nlsPrice', 'max')],
        low=df_candlestick[('nlsPrice', 'min')],
        close=df_candlestick[('nlsPrice', 'last')],
        text=df['nlsShareVolume'],
        increasing_line_color='green',
        decreasing_line_color='red'        
    )


    fig_price = go.Figure(data=candlestick_data)


    fig_price.update_layout(
        xaxis_rangeslider_visible=False,
        title='Price',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price')
    )

    html_out = html.H2(f"Aktuelle Zeit: {current_time} ({n}) ")

    return html_out, fig_price, fig_volume


@app.callback(
    Output("interval", "disabled"),
    Output("start-button", "disabled"),
    Output("stop-button", "disabled"),
    Output("timer-output", "children"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("interval", "disabled"),
    State("timer-output", "children")
)

def start_stop_timer(start_clicks, stop_clicks, interval_disabled, timer_output):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "start-button" in changed_id:
        interval_disabled = False
        start_disabled = True
        stop_disabled = False
    elif "stop-button" in changed_id:
        interval_disabled = True
        start_disabled = False
        stop_disabled = True
    else:
        start_disabled = False
        stop_disabled = True

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return interval_disabled, start_disabled, stop_disabled, current_time



headers = {"Accept-Encoding": "gzip",
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
           }
# -------------------------------------------------
# get_content
# -------------------------------------------------
def get_content(url):
    
    response = requests.get(url, headers=headers).json()
    return response


# Server starten
if __name__ == "__main__":
    app.run_server(debug=True)
