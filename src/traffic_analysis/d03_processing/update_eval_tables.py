import datetime
import pandas as pd 

from traffic_analysis.d05_evaluation.chunk_evaluator import ChunkEvaluator
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def update_eval_tables(db_frame_level_name: str,
                       db_video_level_name: str,
                       params: dict,
                       creds: dict,
                       paths: dict,
                       avg_runtime: float,
                       evaluated_params: dict,
                       # analyser_type: str,
                       return_data=False):
    """Pulls frame level and video level info from specified db names, 
    and writes video level/frame level evaluation results to corresponding 
    video level/frame level PSQL evaluation tables

    Args: 
      db_frame_level_name: name of frame level table (corresponding to analyser_type)
                           to evaluate
      db_video_level_name: name of video level table (corresponding to analyser_type)
                           to evaluate
      params: specified by params.yml 
      creds: specified by creds.yml
      paths: specified by paths.yml
      evaluated_params: dictionary of the parameters which have been altered
      return_data: if true, will return the eval tables
    Raises
    Returns: 
      frame_level_map: pandas DataFrame of frame level statistics (mean average precision)
      video_level_performance: pandas DataFrame of video level summary statistics
      video_level_diff: pandas DataFrame of video level statistics (non-summarize)
    """
    # get xmls from s3
    dl_s3 = DataLoaderBlob(blob_credentials=creds[paths['blob_creds']])
    annotation_xmls = dl_s3.list_objects(prefix=paths['s3_cvat_annotations'])

    # get video_level_tables and frame_level_tables for analyser type from db
    dl_sql = DataLoaderSQL(creds=creds, paths=paths)
    frame_level_df = dl_sql.select_from_table(
        sql=f"SELECT * FROM {db_frame_level_name}"
    )
    video_level_df = dl_sql.select_from_table(
        sql=f"SELECT * FROM {db_video_level_name}"
    )

    # stitch bbox columns back together for frame level
    bboxes = []
    for x, y, w, h in zip(frame_level_df['bbox_x'].values,
                          frame_level_df['bbox_y'].values,
                          frame_level_df['bbox_w'].values,
                          frame_level_df['bbox_h'].values):
        bboxes.append([x, y, w, h])
    frame_level_df['bboxes'] = bboxes
    frame_level_df.drop('bbox_x', axis=1, inplace=True)
    frame_level_df.drop('bbox_y', axis=1, inplace=True)
    frame_level_df.drop('bbox_w', axis=1, inplace=True)
    frame_level_df.drop('bbox_h', axis=1, inplace=True)

    # run evaluation for analyser type
    chunk_evaluator = ChunkEvaluator(annotation_xml_paths=annotation_xmls,
                                     frame_level_df=frame_level_df,
                                     video_level_df=video_level_df,
                                     selected_labels=params["selected_labels"],
                                     video_level_column_order=params["video_level_column_order"],
                                     data_loader_s3=dl_s3
                                     )
    video_level_performance, video_level_diff = chunk_evaluator.evaluate_video_level()
    # prepare for insertion into db

    # video level performance
    video_level_performance = video_level_performance.astype(
        {"n_videos": 'int64'})
    video_level_performance = format_and_add_static_info(video_level_performance, 
                                                         evaluated_params=evaluated_params)

    video_level_performance['avg_analyser_runtime'] = avg_runtime

    # TODO: put this in params
    video_level_performance = video_level_performance[["vehicle_type",
                                                       "stat",
                                                       "bias",
                                                       "MAE",
                                                       "RMSE",
                                                       "sd",
                                                       "n_videos",
                                                       "creation_datetime",
                                                       "avg_analyser_runtime"
                                                       ] + 
                                                       params["eval_params_order"]
                                                       ]
    dl_sql.add_to_sql(df=video_level_performance,
                      table_name=paths["eval_db_video_stats"])

    # video level diff
    video_level_diff = video_level_diff.astype(
        {"camera_id": "object",
         "counts_true": "int64",
         "starts_true": "int64",
         "stops_true": "int64",
         })
    video_level_diff = format_and_add_static_info(video_level_diff, 
                                                  evaluated_params=evaluated_params)

    video_level_diff = video_level_diff[["camera_id",
                                         "video_upload_datetime",
                                         "vehicle_type",
                                         "counts_true",
                                         "starts_true",
                                         "stops_true",
                                         "counts_pred",
                                         "starts_pred",
                                         "stops_pred",
                                         "counts_diff",
                                         "starts_diff",
                                         "stops_diff",
                                         "creation_datetime"
                                         ] + 
                                          params["eval_params_order"]
                                         ]

    dl_sql.add_to_sql(df=video_level_diff,
                      table_name=paths["eval_db_video_diffs"])

    # frame level eval
    frame_level_map = chunk_evaluator.evaluate_frame_level()

    frame_level_map = format_and_add_static_info(frame_level_map, 
                                                  evaluated_params=evaluated_params)

    frame_level_map = frame_level_map[["camera_id",
                                       "video_upload_datetime",
                                       "vehicle_type",
                                       "mean_avg_precision",
                                       "creation_datetime",
                                       ] + 
                                        params["eval_params_order"]
                                       ]

    dl_sql.add_to_sql(df=frame_level_map,
                      table_name=paths["eval_db_frame_stats"])

    if return_data:
        return frame_level_map, video_level_performance, video_level_diff


def format_and_add_static_info(eval_df: pd.DataFrame, 
                               evaluated_params: dict):
    """Creates columns in common to all three eval tables
    """
    eval_df = eval_df.dropna(how='any')
    eval_df['creation_datetime'] = datetime.datetime.now()

    for param_name, param_value in evaluated_params.items():
        eval_df[param_name] = param_value

    return eval_df
