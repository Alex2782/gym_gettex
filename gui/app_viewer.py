from dash import Dash, html, dcc, callback, Output, Input
import json
from dash.exceptions import PreventUpdate
from dash.dependencies import State
import os
import glob
import dash_mantine_components as dmc

app = Dash(__name__)

app.layout = dmc.MantineProvider(
    theme={
        "colorScheme": "dark",
        "primaryColor": "indigo",
        "fontFamily": "'Inter', sans-serif",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Select": {"styles": {"root": {"width": 200}}},
        },
    },
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        html.H1("Dash App mit Mantine", style={"textAlign": "center"}),
        dmc.Stack(
            children=[
                dmc.TextInput(
                    label="Pfad:",
                    id="path-input",
                ),
                dmc.Button("Speichern", id="save-button"),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Label("Dateien:", style={"display": "block"}),
        dmc.Select(
            id="files-dropdown",
            data=[],
            searchable=True,
            nothingFound="No options found",
            style={"width": 200},
        ),
        dcc.Store(id="path-store", storage_type="local"),
    ],
)

@app.callback(
    Output("files-dropdown", "data"),
    Output("files-dropdown", "value"),
    [Input("save-button", "n_clicks")],
    [State("path-input", "value")],
)
def update_files_dropdown(n_clicks, path_input_value):
    if not n_clicks or not path_input_value:
        raise PreventUpdate

    data = []
    value = None

    # Speichern des Pfads im LocalStorage
    path_data = {"path": path_input_value}
    return_value = {"data": data, "value": value, "path_data": path_data}

    # Überprüfe, ob der Pfad gültig ist und Dateien vorhanden sind
    if path_input_value:
        # Suche nach *.feather-Dateien im angegebenen Pfad
        files = glob.glob(os.path.join(path_input_value, "*.feather"))
        filenames = [os.path.basename(file) for file in files]

        data = filenames

        # Überprüfe, ob der aktuell ausgewählte Wert in der Dropdown-Liste enthalten ist
        if path_input_value in filenames:
            value = path_input_value

    return data, value





@app.callback(
    Output("path-store", "data"),
    [Input("save-button", "n_clicks")],
    [State("path-input", "value")],
)
def save_path_to_store(n_clicks, path_input_value):
    if not n_clicks or not path_input_value:
        raise PreventUpdate

    # Speichern des Pfads im LocalStorage
    path_data = {"path": path_input_value}
    return path_data


@app.callback(
    Output("path-input", "value"),
    [Input("path-store", "data")],
)
def load_saved_path_from_store(path_data):
    if not path_data:
        raise PreventUpdate

    # Laden des gespeicherten Werts aus dem LocalStorage
    return path_data.get("path", "")


if __name__ == "__main__":
    app.run_server(debug=True)
