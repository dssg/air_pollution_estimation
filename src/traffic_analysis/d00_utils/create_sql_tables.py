from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def create_sql_tables(drop=False):

    dl = DataLoaderSQL(creds=load_credentials(), paths=load_paths())

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
            vehicle_type VARCHAR(100),
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            counts FLOAT,
            creation_datetime timestamp
        )
        """
    ]

    try:
        dl.open_connection()

        if(drop_commands is not None):
            for command in drop_commands:
                dl.cursor.execute(command)

        for command in commands:
            dl.cursor.execute(command)

        dl.conn.commit()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
        return False

    finally:
        dl.cursor.close()
        if dl.conn:
            dl.conn.close()

    return True
