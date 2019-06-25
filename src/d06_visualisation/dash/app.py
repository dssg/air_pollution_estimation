import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_player as player
from helper import get_cams
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets, hot_reload=True)
server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Traffic Dynamics Statistics</title>

        {%favicon%}
        <link href="https://codepen.io/chriddyp/pen/bWLwgP.css" rel="stylesheet">
        <link href="https://codepen.io/chriddyp/pen/brPBPO.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">

        {%css%}
    </head>
    <body>

        {%app_entry%}
        <footer id="footer">
            {%config%}
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

cams = get_cams()

app.layout = html.Div([
    html.H2("Traffic Dynamics Statistics",
            className="text-center"),
    html.Div(
        [
            html.Label(
                'Cameras in London:',
                className="col-md-2"),
            dcc.Dropdown(
                id="camera_id",
                options=cams,
                className="col-md-6"),
        ],
        id="dropdown_input",
        className="row"),
    html.Div(
        className="row",
        children=[
            html.Div(
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
                ),
                className="col-md-6",
                id='video-outer-container',

            ),
            html.Div(
                className="col-md-6",
                id='statistics')
        ]
    ),
], className="container-fluid")


@app.callback(
    Output("statistics", "children"),
    [
        Input("camera_id", "value")
    ]
)
def update_statistics(camera_id):
    if camera_id:
        children = html.Div(
            [
                html.H3("Frame Level Statistics"),
                html.Div(
                    [
                        dcc.Graph()
                    ],
                    id="frame_level_graphs"

                ),
                html.H3("Video Level Statistics"),
                html.Div(
                    [
                        dcc.Graph()
                    ],
                    id="video_level_graphs"
                )
            ]
        )
        return children

@app.callback(
    Output("video-display", "url"),
    [Input("camera_id", "value")]
)
def update_video_and_statistics(camera_id):
    print(camera_id)
    if camera_id:
        download_url = "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/"
        camera_id = camera_id.replace("JamCams_", "")
        filename = camera_id + ".mp4"
        url = os.path.join(download_url, filename)
        return url


if __name__ == "__main__":
    app.run_server(debug=True)
