from traffic_analysis.d00_utils.load_confs import load_paths, load_parameters, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_sql_tables
from traffic_analysis.d00_utils.upload_setup_data_to_s3 import upload_yolo_weights_to_s3, upload_camera_details_to_s3
from traffic_analysis.d01_data.collect_video_data import download_camera_meta_data

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# get yolo model weights and insert to SQL 
upload_yolo_weights_to_s3(s3_credentials=s3_credentials,
						  bucket_name=paths['bucket_name'],
						  local_dir=paths['temp_setup'],
						  target_dir_on_s3=paths['s3_detection_model']
						  )


# upload camera details S3
upload_camera_details_to_s3(s3_credentials=s3_credentials,
                            bucket_name=paths['bucket_name'],
                            local_dir=paths['setup_data'],
                            target_dir_on_s3=paths['s3_camera_details']
                            )


# create PSQL tables to insert vehicle statistics into 
create_primary_sql_tables(db_vehicle_types_name=paths['db_vehicle_types'],
                  		  db_cameras_name=paths['db_cameras'],
                  		  db_frame_level_name=paths['db_frame_level'], 
                  		  db_video_level_name=paths['db_video_level'],
                  		  drop=True)

# TODO
# put annotated videos in S3, put annotation xmls in right folder 
shutil.rmtree(paths['setup_data'], ignore_errors=True) # cleanup

