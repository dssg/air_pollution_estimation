from traffic_analysis.d00_utils.load_confs import load_paths, load_credentials
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL

paths = load_paths()
creds = load_credentials()

db_host = paths['db_host']
db_name = paths['db_name']
db_user = creds['postgres']['user']
db_pass = creds['postgres']['passphrase']
conn_string = "password=%s user=%s dbname=%s host=%s" % (db_pass, db_user, db_name, db_host)

db_obj = DataLoaderSQL(conn_string)
db_obj.create_tables(True)
