from traffic_analysis.d00_utils.load_confs import load_paths
from traffic_analysis.d00_utils.create_primary_sql_tables import create_primary_sql_tables
from traffic_analysis.d00_utils.upload_yolo_weights_to_s3 import upload_yolo_weights_to_s3
from traffic_analysis.d01_data.collect_video_data import download_camera_meta_data

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]


# get yolo model weights and insert to SQL 
# TODO: debug this
upload_yolo_weights_to_s3(s3_credentials=s3_credentials,
						  bucket_name=paths['bucket_name'],
						  s3_profile=paths['s3_profile'],
						  local_dir=paths['temp_setup_dir'],
						  target_dir_on_s3=paths['s3_detection_model']
						  )

# upload camera details S3
download_camera_meta_data(tfl_camera_api=params['tfl_camera_api'],
                          paths=paths,
                          s3_credentials=s3_credentials
                          )

# run data collection pipeline and insert into S3 bucket to initialize S3 structure 

# create PSQL tables to insert vehicle statistics into 
create_primary_sql_tables(db_vehicle_types_name=paths['db_vehicle_types'],
                  		  db_cameras_name=paths['db_cameras'],
                  		  db_frame_level_name=paths['db_frame_level'], 
                  		  db_video_level_name=paths['db_video_level'],
                  		  drop=True)
