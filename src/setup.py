from traffic_analysis.d00_utils.load_confs import load_paths, load_parameters, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_sql_tables
from traffic_analysis.d00_utils.upload_setup_data_to_s3 import upload_yolo_weights_to_s3
from traffic_analysis.d00_utils.upload_setup_data_to_s3 import upload_annotations_to_s3

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# get yolo model weights and insert to SQL 
upload_yolo_weights_to_s3(s3_credentials=s3_credentials,
                          bucket_name=paths['bucket_name'],
                          local_dir=paths['temp_setup'],
                          target_dir_on_s3=paths['s3_detection_model'])

# create PSQL tables to insert vehicle statistics into 
create_sql_tables(drop=False)

# put annotated videos in S3, put annotation xmls in right folder
upload_annotations_to_s3(s3_credentials=s3_credentials, paths=paths)
