import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player


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


app.layout = html.Div([
    html.H2("Traffic Dynamics Statistics",
            className="text-center"),
    html.Div([
        html.Div(
            className='video-outer-container',
            children=html.Div(
                style={'width': '80%', 'paddingBottom': '56.25%',
                       'position': 'relative'},
                children=player.DashPlayer(
                    id='video-display',
                    style={'position': 'absolute', 'width': '100%',
                           'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                    url='assets/2019-06-20 15:11:25.343799_00001.02151.mp4',
                    controls=True,
                    playing=False,
                    volume=1,
                    width='100%',
                    height='100%'
                )
            )
        )], className="col-md-8"),
    html.Div([
        html.H6("Counts")


    ],
        className="col-md-4",
        id='statistics'),

], className="container-fluid")


if __name__ == "__main__":
    app.run_server(debug=True)
