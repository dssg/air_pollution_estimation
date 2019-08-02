from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def create_sql_tables(drop=False):

    paths = load_paths()

    # create queries
    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE {}, {}, {}, {} CASCADE;".format(paths['db_vehicle_types'],
                                                        paths['db_cameras'],
                                                        paths['db_frame_level'],
                                                        paths['db_video_level'])
        ]

    commands = [
        """
        CREATE TABLE {}(
            id SERIAL,
            vehicle_type VARCHAR(100),
            vehicle_type_id INTEGER PRIMARY KEY
        )
        """.format(paths['db_vehicle_types']),
        """
        CREATE TABLE {}(
            id SERIAL,
            latitude FLOAT, 
            longitude FLOAT,
            borough INTEGER,
            tfl_camera_id INTEGER,
            camera_name VARCHAR(100)
        )
        """.format(paths['db_cameras']),
        """
        CREATE TABLE {}(
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
        """.format(paths['db_frame_level']),

        """
        CREATE TABLE {}(
            vehicle_type VARCHAR(100),
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            counts FLOAT,
            creation_datetime timestamp
        )
        """.format(paths['db_video_level'])
    ]

    # execute queries
    dl = DataLoaderSQL(creds=load_credentials(), paths=paths)

    if drop_commands is not None:
        for command in drop_commands:
            dl.execute_raw_sql_query(sql=command)

    for command in commands:
        dl.execute_raw_sql_query(sql=command)
