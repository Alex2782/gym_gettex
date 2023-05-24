import requests
from sseclient import SSEClient
import json
import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.graph_objs as go

# URL und Header definieren
url = "https://api.boerse-frankfurt.de/v1/data/bid_ask_overview?isin=US88160R1014&mic=XETR"
headers = {'Accept': 'text/event-stream'}

# SSE-Client initialisieren
client = SSEClient(url, headers=headers)

# Dash-App initialisieren
app = dash.Dash(__name__)

# Layout der App
app.layout = html.Div(
    children=[
        html.H1("Bid-Ask Overview"),
        dcc.Input(id="dummy-input", style={"display": "none"}),
        dcc.Graph(id="live-graph")
    ]
)


@app.callback(Output("live-graph", "figure"), [Input("dummy-input", "value")])
def update_graph(dummy_input):

    print ('update_graph')

    for event in client:
        if event.event == 'message':
            decoded_data = event.data.split("data:", 1)[-1].strip()

            #if len(decoded_data) > 0 and decoded_data[0] == '{':
            #    json_data = json.loads(decoded_data)

            json_data = json.loads(decoded_data)
            # Daten verarbeiten und Plots erstellen
            bid_prices = []
            ask_prices = []
            for item in json_data["data"]:
                bid_prices.append(item["bidPrice"])
                ask_prices.append(item["askPrice"])

            trace1 = go.Scatter(x=list(range(len(bid_prices))), y=bid_prices, mode="lines", name="Bid Prices")
            trace2 = go.Scatter(x=list(range(len(ask_prices))), y=ask_prices, mode="lines", name="Ask Prices")
            data = [trace1, trace2]
            layout = go.Layout(title="Bid-Ask Prices", xaxis=dict(title="Time"), yaxis=dict(title="Price"))
            fig = go.Figure(data=data, layout=layout)
            return fig


if __name__ == "__main__":
    app.run_server(debug=True)
