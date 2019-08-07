from traffic_analysis.d05_evaluation.chunk_evaluator import ChunkEvaluator
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
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
	# get xmls
    dl_s3 = DataLoaderS3(s3_credentials=creds[paths['s3_creds']],
                        bucket_name =  paths['bucket_name'])
	annotation_xmls = dl_s3.list_objects(prefix = 'ref/annotations/cvat/')

	# get video_level_tables and frame_level_tables for analyser type from db 

	# run evaluation for analyser type 
	chunk_evaluator = ChunkEvaluator(annotation_xml_paths=annotation_xmls,
						             frame_level_df=frame_level_df,
	             					 video_level_df=video_level_df,
	             					 params=params,
	             					 data_loader_s3 = dl_s3
						             )
	video_level_performance, video_level_diff = chunk_evaluator.evaluate_video_level()
	frame_level_map = chunk_evaluator.evaluate_frame_level()

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

	db_obj = DataLoaderSQL(creds=creds, paths=paths)
	for db_name, df in eval_dfs.items(): 
		df.dropna(how='any',inplace=True)
		df['analyser'] = analyser_type
 		df['creation_datetime'] = datetime.datetime.now()
		# append to sql table 
    	db_obj.add_to_sql(df=df, table_name=db_name)

	if return_data: 
		return frame_level_map, video_level_performance, video_level_diff

