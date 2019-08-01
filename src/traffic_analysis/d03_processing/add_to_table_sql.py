import sqlalchemy

from traffic_analysis.d00_utils.data_access import db

def add_to_table_sql(df, table, creds, paths):

    db_host = paths['db_host']
    db_name = paths['db_name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn_string = "password=%s user=%s dbname=%s host=%s" % (
        db_pass, db_user, db_name, db_host)

    db_obj = db(conn_string=conn_string)
    db_obj.save_data(df=df, table_name=table)

    return
