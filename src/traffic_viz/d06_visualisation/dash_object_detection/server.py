import dash

app = dash.Dash(__name__, hot_reload=True)
server = app.server
app.scripts.config.serve_locally = True
app.config["suppress_callback_exceptions"] = True
