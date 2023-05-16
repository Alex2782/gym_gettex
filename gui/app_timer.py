import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
from dash.dependencies import Output, Input
import datetime

# Dash-Anwendung erstellen
app = dash.Dash(__name__)

# Layout erstellen
app.layout = html.Div(
    children=[
        html.H1("Timer Beispiel"),
        html.Div(id="timer-output"),
        dcc.Interval(id="timer", interval=1000, n_intervals=0),  # Intervall-Timer-Komponente
    ]
)

# Callback-Funktion für Timer
@app.callback(
    Output("timer-output", "children"),
    Input("timer", "n_intervals")
)
def update_timer(n):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return html.H2(f"Aktuelle Zeit: {current_time} ({n})")

# Server starten
if __name__ == "__main__":
    app.run_server(debug=True)
