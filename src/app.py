from datetime import datetime as dt

import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from traffic_viz.d06_visualisation.helper import (
    app_params, 
    get_cams, 
    get_vehicle_types, 
    load_camera_statistics,
    load_vehicle_type_statistics)
from traffic_viz.d06_visualisation.server import app


DEBUG = app_params["debug"]
TFL_BASE_URL = app_params["tfl_jamcams_website"]

cams = get_cams()


# Main App
app.layout = html.Div(
    children=[
        html.Div(
            id="top-bar",
            className="row",
            style={"backgroundColor": "#fa4f56", "height": "5px"},
        ),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="eight columns",
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "flex": 1,
                        "height": "calc(100vh - 5px)",
                        "backgroundColor": "#F2F2F2",
                        "overflow-y": "scroll",
                        "marginLeft": "0px",
                        "justifyContent": "flex-start",
                        "alignItems": "center",
                    },
                    children=[
                        html.Div(
                            id="header-section",
                            children=[html.H4("Traffic Dynamics Explorer")],
                        ),
                        html.Div(
                            className="video-outer-container",
                            children=[
                                html.H6("Select a junction to view footage."),
                                html.Div(
                                    style={
                                        "width": "100%",
                                        "paddingBottom": "56.25%",
                                        "position": "relative",
                                    },
                                    children=player.DashPlayer(
                                        id="video-display",
                                        style={
                                            "position": "absolute",
                                            "width": "100%",
                                            "height": "100%",
                                            "top": "0",
                                            "left": "0",
                                            "bottom": "0",
                                            "right": "0",
                                        },
                                        controls=True,
                                        playing=False,
                                        volume=1,
                                        width="100%",
                                        height="100%",
                                    ),
                                )
                            ],
                        ),
                        html.Div(
                            className="control-section",
                            children=[
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(
                                            children=["Junction Selection:"],
                                            style={"width": "40%"},
                                        ),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=[
                                                {"label": v, "value": k}
                                                for k, v in cams.items()
                                            ],
                                            clearable=False,
                                            style={"width": "60%"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(
                                            children=["Objects:"],
                                            style={"width": "40%"},
                                        ),
                                        dcc.Dropdown(
                                            id="dropdown-vehicle-types",
                                            clearable=False,
                                            style={"width": "60%"},
                                            multi=True,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    [
                                        html.Div("Date Range:", style={
                                                 "width": "40%"}),
                                        html.Div(
                                            [
                                                dcc.DatePickerRange(
                                                    id="daterange",
                                                    min_date_allowed=app_params[
                                                        "min_date_allowed"
                                                    ],
                                                    max_date_allowed=dt.now(),
                                                    end_date=dt.now(),
                                                    clearable=True,
                                                    start_date=app_params["start_date"],
                                                )
                                            ],
                                            className="",
                                        ),
                                    ],
                                    className="control-element",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="four columns",
                    style={
                        "height": "calc(100vh - 5px)",
                        "overflow-y": "scroll",
                        "marginLeft": "1%",
                        "display": "flex",
                        "backgroundColor": "#F9F9F9",
                        "flexDirection": "column",
                    },
                    children=[
                        html.Div(
                            className="img-container",
                            children=html.Img(
                                style={"height": "100%", "margin": "2px"},
                                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
                            ),
                        ),
                        html.Div(id="div-visual-mode"),
                    ],
                ),
            ],
        ),
    ]
)


# Footage Selection
@app.callback(Output("video-display", "url"),
              [Input("dropdown-footage-selection", "value")])
def select_footage(footage):
    # Find desired footage and update player video
    if footage:
        footage = footage.replace("JamCams_", "")
        filename = footage + ".mp4"
        url = TFL_BASE_URL + filename
        print(url)
        return url


@app.callback(
    Output("dropdown-vehicle-types", "options"),
    [Input("dropdown-footage-selection", "value")],
)
def update_objects(camera_id):
    global df
    camera_id = transform_camera_id(camera_id)
    df = load_camera_statistics(camera_id)
    if df.empty:
        print("Empty")
        return []
    if camera_id:
        vehicle_types = get_vehicle_types()
        options = [{"label": obj, "value": obj} for obj in vehicle_types]
        return options


def transform_camera_id(camera_id):
    if camera_id:
        camera_id = camera_id.replace("JamCams_", "")
    return camera_id


@app.callback(
    Output("trend-graph", "figure"),
    [
        Input("dropdown-vehicle-types", "value"),
        Input("dropdown-footage-selection", "value"),
        Input("daterange", "start_date"),
        Input("daterange", "end_date"),
    ],
)
def update_trend_graph(vehicle_types, camera_id, start_date, end_date):
    global df
    if not camera_id:
        return
    camera_name = cams[camera_id]
    title = camera_name
    camera_id = transform_camera_id(camera_id)
    df = load_camera_statistics(camera_id)
    if df.empty or not vehicle_types:
        return {}
    dfs = {
        obj: load_vehicle_type_statistics(df, obj, start_date, end_date)
        for obj in vehicle_types
    }
    print(dfs)
    data = []
    for obj, df_stats in dfs.items():
        print(df_stats.head())
        data.append(
            go.Scatter(
                x=df_stats["video_upload_datetime"],
                y=df_stats["counts"],
                name=obj,
                mode="lines"))

    figure = {
        "data": data,
        "layout": go.Layout(
            showlegend=True,
            title=title,
            xaxis={
                "title": "Datetime"},
            yaxis=go.layout.YAxis(
                title="Count of vehicle types [#]",
                automargin=True),
            hovermode="closest",
            autosize=False,
        ),
    }
    print(figure)
    return figure


@app.callback(
    Output("div-visual-mode", "children"),
    [Input("dropdown-footage-selection", "value")],
)
def update_output(camera_id):
    print(camera_id)
    if camera_id:
        return [
            dcc.Interval(
                id="interval-visual-mode",
                interval=700,
                n_intervals=0),
            html.Div(
                children=[
                    html.P(
                        children="Trend of vehicle types in video",
                        className="plot-title",
                    ),
                    dcc.Graph(
                        id="trend-graph", style={"height": "45vh", "width": "100%"}
                    ),
                    # html.P(children="Object Count",
                    #        className='plot-title'),
                    # dcc.Graph(
                    #     id="pie-object-count",
                    #     style={'height': '40vh', 'width': '100%'}
                    # )
                ]
            ),
        ]
    else:
        return []


# Running the server
if __name__ == "__main__":
    app.run_server(debug=DEBUG, host="0.0.0.0")
