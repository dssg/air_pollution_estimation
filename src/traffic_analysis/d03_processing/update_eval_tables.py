from traffic_analysis.d05_evaluation.chunk_evaluator import ChunkEvaluator
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
import pandas as pd 

from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths

params = load_parameters()
creds = load_credentials()
paths = load_paths()

s3_credentials = creds[paths['s3_creds']]
bucket_name = paths['bucket_name']


def update_eval_tables(db_frame_level_name, 
                       db_video_level_name,
                       params: dict,
                       creds: dict, 
                       paths: dict,
                       analyser_type: str,
                       return_data=False): 
    """
    """
    # get xmls from s3
    dl_s3 = DataLoaderS3(s3_credentials=creds[paths['s3_creds']],
                        bucket_name =  paths['bucket_name'])
    annotation_xmls = dl_s3.list_objects(prefix = paths['s3_cvat_annotations'])

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
                                     params=params,
                                     data_loader_s3 = dl_s3
                                     )
    video_level_performance, video_level_diff = chunk_evaluator.evaluate_video_level()
    frame_level_map = chunk_evaluator.evaluate_frame_level()

    # prepare for insertion into db
    video_level_performance = video_level_performance.astype(
        {"n_videos": 'int64'})

    video_level_diff = video_level_diff.astype(
         {"camera_id": "object",
          "counts_true": "int64",
          "starts_true": "int64",
          "stops_true": "int64",
          })

    eval_dfs = {"eval_video_performance": video_level_performance, 
                "eval_video_diffs": video_level_diff, 
                "eval_frame_stats": frame_level_map}

    for db_name, df in eval_dfs.items(): 
        df.dropna(how='any',inplace=True)
        df['analyser'] = analyser_type
        df['creation_datetime'] = datetime.datetime.now()
        # append to sql db
        dl_sql.add_to_sql(df=df, table_name=db_name)

    if return_data: 
        return frame_level_map, video_level_performance, video_level_diff
