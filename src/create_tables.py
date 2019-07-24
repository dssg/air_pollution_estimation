from traffic_analysis.d00_utils.load_confs import load_paths, load_credentials
from traffic_analysis.d00_utils.data_access import db

paths = load_paths()
creds = load_credentials()

db_host = paths['db_host']
db_name = paths['db_name']
db_user = creds['postgres']['user']
db_pass = creds['postgres']['passphrase']
conn_string = "password=" + db_pass + " user=" + db_user + " dbname=" + db_name + " host=" + db_host

db_obj = db(conn_string)
db_obj.create_tables(True)
