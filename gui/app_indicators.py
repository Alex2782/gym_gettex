import dash
from dash import dcc
from dash import html
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np

stock_data = None
last_symbol = None

def get_stock_data(symbol):
    stock_data = yf.download(symbol, start='2021-01-01', end='2023-06-01')
    return stock_data


# App initialisieren
app = dash.Dash(__name__)

# Dropdown-Optionen für Aktienliste
stock_options = [
    {'label': 'Tesla', 'value': 'TSLA'},
    {'label': 'Apple', 'value': 'AAPL'},
    {'label': 'Alphabet (Google)', 'value': 'GOOGL'},
    {'label': 'Microsoft', 'value': 'MSFT'},
]

# Dropdown-Optionen für Indikatoren
indicator_options = [
    {'label': 'SMA 50', 'value': 'sma50'},
    {'label': 'SMA 200', 'value': 'sma200'},
    {'label': 'EMA 12', 'value': 'ema12'},
    {'label': 'EMA 24', 'value': 'ema24'},
    {'label': 'Weighted Moving Average (WMA)', 'value': 'wma'},
    {'label': 'Hull Moving Average (HMA)', 'value': 'hma'},
    {'label': 'Triple Exponential Moving Average (TEMA)', 'value': 'tema'},
    {'label': 'Volume Weighted Average Price (VWAP)', 'value': 'vwap'},    
]

# Dash-Layout erstellen
app.layout = html.Div([
    html.H1('Aktiendaten mit Indikatoren'),
    
    html.Label('Aktie auswählen'),
    dcc.Dropdown(
        id='stock-dropdown',
        options=stock_options,
        value='TSLA'
    ),
    
    html.Label('Indikatoren auswählen'),
    dcc.Dropdown(
        id='indicator-dropdown',
        options=indicator_options,
        value=['sma50'],
        multi=True
    ),
    
    dcc.Graph(id='stock-chart')
])

# Callback für Diagramm-Update
@app.callback(
    Output('stock-chart', 'figure'),
    Input('stock-dropdown', 'value'),
    Input('indicator-dropdown', 'value'))
def update_stock_chart(stock_symbol, indicators):

    global stock_data, last_symbol

    # Aktualisiere Aktiendaten nur, wenn 'stock-dropdown' verändert wurde
    if stock_data is None or stock_symbol != last_symbol:
        stock_data = get_stock_data(stock_symbol)
        last_symbol = stock_symbol

    figure = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Aktienkurs'
    )])

    # Layout anpassen, um die Höhe zu konfigurieren
    figure.update_layout(height=800, 
                         xaxis_rangeslider_visible=False,
                         )    

    for indicator in indicators:
        if indicator == 'sma50':
            sma = stock_data['Close'].rolling(window=50).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=sma, name='SMA 50'))
        elif indicator == 'sma200':
            sma = stock_data['Close'].rolling(window=200).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=sma, name='SMA 200'))
        elif indicator == 'ema12':
            ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=ema, name='EMA 12'))
        elif indicator == 'ema24':
            ema = stock_data['Close'].ewm(span=24, adjust=False).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=ema, name='EMA 24'))
        elif indicator == 'wma':
            wma = stock_data['Close'].rolling(window=50).apply(lambda x: np.dot(x, np.arange(1, 51)) / np.sum(np.arange(1, 51)))
            figure.add_trace(go.Scatter(x=stock_data.index, y=wma, name='WMA'))
        elif indicator == 'hma':
            hma = 2 * stock_data['Close'].rolling(window=int(len(stock_data)/2)).mean() - stock_data['Close'].rolling(window=len(stock_data)).mean().rolling(window=int(np.sqrt(len(stock_data)))).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=hma, name='HMA'))
        elif indicator == 'tema':
            tema = stock_data['Close'].ewm(span=12, adjust=False).mean().ewm(span=12, adjust=False).mean().ewm(span=12, adjust=False).mean()
            figure.add_trace(go.Scatter(x=stock_data.index, y=tema, name='TEMA'))
        elif indicator == 'vwap':
            vwap = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
            figure.add_trace(go.Scatter(x=stock_data.index, y=vwap, name='VWAP'))
 
    return figure

# App ausführen
if __name__ == '__main__':
    app.run_server(debug=True)
