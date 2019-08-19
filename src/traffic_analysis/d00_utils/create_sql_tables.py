from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def create_sql_tables(drop=False):

    paths = load_paths()

    # create queries
    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE {}, {} CASCADE;".format(paths['db_frame_level'],
                                                paths['db_video_level'])
        ]

    commands = [
        """
        CREATE TABLE {}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            frame_id SMALLINT,
            vehicle_id SMALLINT,
            vehicle_type VARCHAR(20),
            confidence REAL,
            box_x SMALLINT,
            box_y SMALLINT,
            box_w SMALLINT,
            box_h SMALLINT,
            creation_datetime timestamp
        )
        """.format(paths['db_frame_level']),

        """
        CREATE TABLE {}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            vehicle_type VARCHAR(20),
            counts REAL,
            stops REAL,
            starts REAL,
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
