import psycopg2

from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL

def create_sql_tables(drop=False):

    dl = DataLoaderSQL(creds=load_credentials(), paths=load_paths())
    dl.open_connection()
    conn = dl.conn
    cursor = dl.cursor

    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE vehicle_types, cameras, frame_stats, video_stats CASCADE;"
        ]

    commands = [
        """
        CREATE TABLE vehicle_types(
            id SERIAL,
            vehicle_type VARCHAR(255),
            vehicle_type_id INTEGER PRIMARY KEY
        )
        """,
        """
        CREATE TABLE cameras(
            id SERIAL,
            latitude FLOAT, 
            longitude FLOAT,
            borough INTEGER,
            tfl_camera_id INTEGER,
            camera_name VARCHAR(500)
        )
        """,
        """
        CREATE TABLE frame_stats (
            camera_id VARCHAR(255),
            frame_id INTEGER,
            datetime timestamp,
            obj_classification VARCHAR(255),
            confidence FLOAT,
            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER
        )
        """,

        """
        CREATE TABLE video_stats (
            counts FLOAT,
            vehicle_type VARCHAR(255),
            camera_id VARCHAR(255),
            datetime timestamp,
            starts FLOAT,
            stops FLOAT
        )
        """
    ]
    try:

        if(drop_commands is not None):
            for command in drop_commands:
                cursor.execute(command)

        for command in commands:
            cursor.execute(command)

        conn.commit()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
        return False

    finally:
        cursor.close()
        if conn:
            conn.close()
    return True
