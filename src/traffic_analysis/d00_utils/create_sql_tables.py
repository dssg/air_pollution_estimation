from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def create_primary_sql_tables(db_frame_level_name: str, 
                              db_video_level_name: str,
                              db_vehicle_types_name: str = None,
                              db_cameras_name: str = None,
                              drop=False
                              ):

    if (db_vehicle_types_name is not None) and (db_cameras_name is not None): 
        handle_vehicles_cameras_tables = True 

    # create queries
    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE {}, {} CASCADE;".format(db_frame_level_name,
                                                db_video_level_name
                                                )
        ]

        if handle_vehicles_cameras_tables: 
            drop_commands.append(
            "DROP TABLE {}, {} CASCADE;".format(db_vehicle_types_name,
                                                db_cameras_name
                                                )
            )


    commands = [
        """
        CREATE TABLE {}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            frame_id INTEGER,
            vehicle_id INTEGER,
            vehicle_type VARCHAR(100),
            confidence FLOAT,
            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER,
            creation_datetime timestamp
        )
        """.format(db_frame_level_name),

        """
        CREATE TABLE {}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            vehicle_type VARCHAR(100),
            counts FLOAT,
            stops FLOAT,
            starts FLOAT,
            creation_datetime timestamp
        )
        """.format(db_video_level_name)
    ]

    if handle_vehicles_cameras_tables: 
        commands += [
            """
            CREATE TABLE {}(
                id SERIAL,
                vehicle_type VARCHAR(100),
                vehicle_type_id INTEGER PRIMARY KEY
            )
            """.format(db_vehicle_types_name),
            """
            CREATE TABLE {}(
                id SERIAL,
                latitude FLOAT, 
                longitude FLOAT,
                borough INTEGER,
                tfl_camera_id INTEGER,
                camera_name VARCHAR(100)
            )
            """.format(db_cameras_name),
        ]

    # execute queries
    dl = DataLoaderSQL(creds=load_credentials(), paths=load_paths())

    if drop_commands is not None:
        for command in drop_commands:
            dl.execute_raw_sql_query(sql=command)

    for command in commands:
        dl.execute_raw_sql_query(sql=command)


def create_eval_sql_tables(creds: dict,
                           paths: dict, 
                           drop=False):
    # create queries
    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE {}, {}, {} CASCADE;".format(paths["eval_db_video_stats"],
                                                paths["eval_db_video_diffs"],
                                                paths["eval_db_frame_stats"]
                                                )
        ]

    commands = [
        f"""
        CREATE TABLE {paths["eval_db_video_stats"]}(
            vehicle_type VARCHAR(100),
            stat VARCHAR(20),
            bias FLOAT,
            MAE FLOAT,
            RMSE FLOAT,
            sd FLOAT, 
            n_videos INT,
            creation_datetime timestamp,
            analyser_type VARCHAR(50)
        )
        """,

        f"""
        CREATE TABLE {paths["eval_db_video_diffs"]}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            vehicle_type VARCHAR(100),
            count_true INT,
            starts_true INT,
            stops_true INT,
            counts_pred FLOAT, 
            starts_pred FLOAT,
            stops_pred FLOAT,
            counts_diff FLOAT, 
            starts_diff FLOAT,
            stops_diff FLOAT,
            creation_datetime timestamp,
            analyser_type VARCHAR(50)

        )
        """,

        f"""
        CREATE TABLE {paths["eval_db_frame_stats"]}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            vehicle_type VARCHAR(100),
            mean_avg_precision FLOAT,
            creation_datetime timestamp,
            analyser_type VARCHAR(50)

        )
        """        
    ]

    # execute queries
    dl = DataLoaderSQL(creds=load_credentials(), paths=load_paths())

    if drop_commands is not None:
        for command in drop_commands:
            dl.execute_raw_sql_query(sql=command)

    for command in commands:
        dl.execute_raw_sql_query(sql=command)
