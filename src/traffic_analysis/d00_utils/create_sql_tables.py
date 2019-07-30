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
            vehicle_type VARCHAR(100),
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
            camera_name VARCHAR(100)
        )
        """,
        """
        CREATE TABLE frame_stats (
            camera_id VARCHAR(20),
            frame_id INTEGER,
            video_upload_datetime timestamp,
            obj_classification VARCHAR(100),
            confidence FLOAT,
            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER,
            creation_datetime timestamp
        )
        """,

        """
        CREATE TABLE video_stats (
            counts FLOAT,
            vehicle_type VARCHAR(100),
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            starts FLOAT,
            stops FLOAT,
            creation_datetime timestamp
        )
        """
    ]

    dl.execute_raw_sql_query(commands)

    return True
