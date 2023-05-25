import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import yfinance as yf

app = dash.Dash(__name__)

# Dropdown-Menü für die Aktienauswahl
stock_dropdown = dcc.Dropdown(
    id='stock-dropdown',
    options=[
        {'label': 'Tesla', 'value': 'TSLA'},
        {'label': 'Apple', 'value': 'AAPL'},
        {'label': 'Alphabet', 'value': 'GOOGL'},
        {'label': 'Microsoft', 'value': 'MSFT'}
    ],
    value='TSLA'
)

# Div-Element für die Anzeige der ausgewählten Aktie
stock_info_div = html.Div(id='stock-info')

# Layout der Dash-App
app.layout = html.Div([
    html.H1('Aktieninformationen'),
    stock_dropdown,
    stock_info_div
])

# Callback für die Aktualisierung der Aktieninformationen
@app.callback(
    Output('stock-info', 'children'),
    Input('stock-dropdown', 'value')
)
def update_stock_info(stock_symbol):
    # Aktiendaten abrufen
    stock = yf.Ticker(stock_symbol)

    # Historische Kursdaten abrufen
    historical_data = stock.history(period='1y')

    # Dividendendaten abrufen
    dividends = stock.dividends

    # Inhalte für die Anzeige erstellen
    content = html.Div([
        html.H2(f'Aktieninformationen für {stock_symbol}'),
        html.H3('Historische Kursdaten'),
        dcc.Graph(
            figure={
                'data': [
                    {'x': historical_data.index, 'y': historical_data['Close'], 'type': 'line', 'name': 'Schlusskurs'}
                ],
                'layout': {
                    'title': 'Historische Kursdaten'
                }
            }
        ),
        html.H3('Dividendendaten'),
        html.Pre(dividends.to_string())
    ])

    return content

if __name__ == '__main__':
    app.run_server(debug=True)
