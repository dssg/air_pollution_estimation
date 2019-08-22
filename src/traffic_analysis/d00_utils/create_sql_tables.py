from traffic_analysis.d00_utils.load_confs import load_credentials, load_paths
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def create_primary_sql_tables(db_frame_level_name: str, 
                              db_video_level_name: str,
                              drop=False
                              ):
    """Create PSQL tables for traffic analyser objects to append to 
    """
    # create queries
    drop_commands = None
    if drop:
        drop_commands = [
            "DROP TABLE {}, {} CASCADE;".format(db_frame_level_name,
                                                db_video_level_name
                                                )
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
        """.format(db_frame_level_name),

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
        """.format(db_video_level_name)
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
    """Create PSQL table to append evaluation results to 
    """
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
            avg_analyser_runtime FLOAT,
            tracker_type VARCHAR(20),
            detection_model VARCHAR(20),
            detection_frequency INT,
            detection_iou_threshold FLOAT,
            stop_start_iou_threshold FLOAT
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
            tracker_type VARCHAR(20),
            detection_model VARCHAR(20),
            detection_frequency INT,
            detection_iou_threshold FLOAT,
            stop_start_iou_threshold FLOAT
        )
        """,

        f"""
        CREATE TABLE {paths["eval_db_frame_stats"]}(
            camera_id VARCHAR(20),
            video_upload_datetime timestamp,
            vehicle_type VARCHAR(100),
            mean_avg_precision FLOAT,
            creation_datetime timestamp,
            tracker_type VARCHAR(20),
            detection_model VARCHAR(20),
            detection_frequency INT,
            detection_iou_threshold FLOAT,
            stop_start_iou_threshold FLOAT
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
