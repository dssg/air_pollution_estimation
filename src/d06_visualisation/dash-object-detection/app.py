from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from helper import get_cams, load_data, load_camera_statistics, load_objects, load_object_statistics
import os
import sys
import time
from datetime import datetime as dt


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', '..', 'd04_modelling'))


DEBUG = True
FRAMERATE = 24.0
TFL_BASE_URL = "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/"

app = dash.Dash(
    __name__, hot_reload=True)
server = app.server

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True
cams = get_cams()
yolo_models = ["yolov3_tiny"]


def markdown_popup():
    return html.Div(
        id='markdown',
        className="model",
        style={'display': 'none'},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={'border': 'none', 'height': '100%'}
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=[dcc.Markdown(
                            children=dedent(
                                '''
                                # What am I looking at?

                                This app enhances visualization of objects detected using state-of-the-art Mobile Vision Neural Networks.
                                Most user generated videos are dynamic and fast-paced, which might be hard to interpret. A confidence
                                heatmap stays consistent through the video and intuitively displays the model predictions. The pie chart
                                lets you interpret how the object classes are divided, which is useful when analyzing videos with numerous
                                and differing objects.

                                # More about this dash app

                                The purpose of this demo is to explore alternative visualization methods for Object Detection. Therefore,
                                the visualizations, predictions and videos are not generated in real time, but done beforehand. To read
                                more about it, please visit the [project repo](https://github.com/plotly/dash-object-detection).

                                '''
                            ))
                        ]
                    )
                ]
            )
        )
    )


# Main App

app.layout = html.Div(
    children=[
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        html.Div(
            className='container',
            children=[
                html.Div(
                    id='left-side-column',
                    className='eight columns',
                    style={'display': 'flex',
                           'flexDirection': 'column',
                           'flex': 1,
                           'height': 'calc(100vh - 5px)',
                           'backgroundColor': '#F2F2F2',
                           'overflow-y': 'scroll',
                           'marginLeft': '0px',
                           'justifyContent': 'flex-start',
                           'alignItems': 'center'},
                    children=[
                        html.Div(
                            id='header-section',
                            children=[
                                html.H4(
                                    'Traffic Dynamics Explorer'
                                ),
                                html.P(
                                    'To get started, select a footage you want to view, and choose the minimum confidence threshold. Then, you can start playing the video, and the visualization will '
                                    'be displayed depending on the current time.'
                                ),
                            ]
                        ),
                        html.Div(
                            className='video-outer-container',
                            children=html.Div(
                                style={'width': '100%', 'paddingBottom': '56.25%',
                                       'position': 'relative'},
                                children=player.DashPlayer(
                                    id='video-display',
                                    style={'position': 'absolute', 'width': '100%',
                                           'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                                    controls=True,
                                    playing=False,
                                    volume=1,
                                    width='100%',
                                    height='100%'
                                )
                            )
                        ),
                        html.Div(
                            className='control-section',
                            children=[
                                # html.Div(
                                #     className='control-element',
                                #     children=[
                                #         html.Div(children=["Minimum Confidence Threshold:"], style={
                                #             'width': '40%'}),
                                #         html.Div(dcc.Slider(
                                #             id='slider-minimum-confidence-threshold',
                                #             min=20,
                                #             max=80,
                                #             marks={
                                #                 i: f'{i}%' for i in range(20, 81, 10)},
                                #             value=30,
                                #             updatemode='drag'
                                #         ), style={'width': '60%'})
                                #     ]
                                # ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Footage Selection:"], style={
                                            'width': '40%'}),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=cams,
                                            value=cams[0]["value"],
                                            clearable=False,
                                            style={'width': '60%'}
                                        )
                                    ]
                                ),
                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Objects:"], style={
                                            'width': '40%'}),
                                        dcc.Dropdown(
                                            id="dropdown-objects", clearable=False,
                                            style={'width': '60%'},
                                            multi=True
                                        )
                                    ]
                                ),
                                html.Div([
                                    html.Div(
                                        'Date Range:', style={
                                            'width': '40%'}),
                                    html.Div([
                                        dcc.DatePickerRange(
                                            id='daterange',
                                            min_date_allowed=dt(
                                                2017, 1, 1),
                                            max_date_allowed=dt.now(),
                                            end_date=dt.now(),
                                            clearable=True,
                                            start_date=dt(
                                                2017, 1, 1),
                                        ),
                                    ], className=''),
                                ], className='control-element'),

                                # html.Div(
                                #     className='control-element',
                                #     children=[
                                #         html.Div(children=["Yolo Model:"], style={
                                #             'width': '40%'}),
                                #         dcc.Dropdown(
                                #             id="dropdown-yolo-model",
                                #             options=[
                                #                 {"label": model, "value": model}
                                #                 for model in yolo_models],
                                #             value=yolo_models[0],
                                #             clearable=False,
                                #             style={'width': '60%'}
                                #         )
                                #     ]
                                # ),
                                # html.Button(
                                #     "Detect Objects", id="detect-objects-button", n_clicks=0)

                            ]
                        )
                    ]
                ),
                html.Div(
                    id='right-side-column',
                    className='four columns',
                    style={
                        'height': 'calc(100vh - 5px)',
                        'overflow-y': 'scroll',
                        'marginLeft': '1%',
                        'display': 'flex',
                        'backgroundColor': '#F9F9F9',
                        'flexDirection': 'column'
                    },
                    children=[
                        html.Div(
                            className='img-container',
                            children=html.Img(
                                style={'height': '100%', 'margin': '2px'},
                                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png")
                        ),
                        html.Div(id="div-visual-mode"),
                    ]
                )]),
        markdown_popup()
    ]
)


# Data Loading
@app.server.before_first_request
def load_all_footage():

    pass


# Footage Selection
@app.callback(
    Output("video-display", "url"),
    [
        Input('dropdown-footage-selection', 'value'),
    ],
)
def select_footage(footage):
    # Find desired footage and update player video
    footage = footage.replace("JamCams_", "")
    filename = footage + ".mp4"
    url = TFL_BASE_URL + filename
    print(url)
    return url


@app.callback(
    Output("dropdown-objects", "options"),
    [
        Input('dropdown-footage-selection', 'value')
    ]
)
def update_objects(camera_id):
    global df
    camera_id = transform_camera_id(camera_id)
    df = load_camera_statistics(camera_id)
    if df.empty:
        print("Empty")
        return []
    if camera_id:
        objects = load_objects(df)
        options = [
            {"label": obj, "value": obj}
            for obj in objects]
        return options


def transform_camera_id(camera_id):
    camera_id = camera_id.replace("JamCams_", "")
    return camera_id


@app.callback(
    Output("trend-graph", "figure"),
    [
        Input("dropdown-objects", "value"),
        Input("dropdown-footage-selection", "value")
    ]
)
def update_trend_graph(objects, camera_id):
    global df
    camera_id = transform_camera_id(camera_id)
    df = load_camera_statistics(camera_id)
    if df.empty or not objects:
        return {}
    dfs = {obj: load_object_statistics(df, obj) for obj in objects}
    print(dfs)
    data = []
    for obj, df_stats in dfs.items():
        # data.append(
        #     go.Scatter(
        #         name='Upper Bound',
        #         x=df_stats.index,
        #         y=df_stats["mean"]+df_stats["std"],
        #         mode='lines',
        #         marker=dict(color="#444"),
        #         line=dict(width=0),
        #         showlegend=False,
        #         fillcolor='rgba(68, 68, 68, 0.3)',
        #         fill='tonexty'
        #     ))
        data.append(
            go.Scatter(
                x=df_stats.index,
                y=df_stats["mean"],
                name=obj,
                mode='lines',
                # line=dict(color='rgb(31, 119, 180)'),
                # fillcolor='rgba(68, 68, 68, 0.3)',
                # fill='tonexty'
            ))
        # data.append(
        #     go.Scatter(
        #         name='Lower Bound',
        #         x=df_stats.index,
        #         y=df_stats["mean"]-df_stats["std"],
        #         showlegend=False,
        #         marker=dict(color="#444"),
        #         line=dict(width=0),
        #         fillcolor='rgba(68, 68, 68, 0.3)',
        #         mode='lines'
        #     ))

    # data = [
    #     go.Scatter(
    #         x=df_stats.index,
    #         y=df_stats["mean"],
    #         name=obj,
    #         mode='lines'
    #     )
    #     for obj, df_stats in dfs.items()
    # ]
    print(data)
    figure = {
        'data': data,
        'layout': go.Layout(
            showlegend=True,
            # title=title,
            xaxis={'title': "Datetime"},
            yaxis=go.layout.YAxis(
                title='Count of objects [#]', automargin=True
            ),
            hovermode='closest',
            # updatemenus=updatemenus,
            # annotations=annotations,
            autosize=False,
        )
    }
    print(figure)
    return figure


@app.callback(
    Output("div-visual-mode", "children"),
    [
        Input("dropdown-footage-selection", "value")
    ])
def update_output(camera_id):
    if camera_id:
        return [
            dcc.Interval(
                id="interval-visual-mode",
                interval=700,
                n_intervals=0
            ),
            html.Div(
                children=[
                    html.P(children="Trend of objects in video",
                           className='plot-title'),
                    dcc.Graph(
                        id="trend-graph",
                        style={'height': '45vh', 'width': '100%'}),
                    # html.P(children="Object Count",
                    #        className='plot-title'),
                    # dcc.Graph(
                    #     id="pie-object-count",
                    #     style={'height': '40vh', 'width': '100%'}
                    # )
                ]
            )
        ]
    else:
        return []


# Running the server
if __name__ == '__main__':
    app.run_server(debug=DEBUG, host='0.0.0.0')
