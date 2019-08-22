import os
import sys
# src_dir = os.path.join(os.getcwd(), '..', 'src')
# sys.path.append(src_dir)
# ​
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d04_modelling.transfer_learning.convert_darknet_to_tensorflow import yolov3_darknet_to_tensorflow

params = load_parameters()
paths = load_paths()
creds = load_credentials()

s3_credentials = creds[paths['s3_creds']]
params['detection_model'] = "yolov3_tf"  # yolov3_opencv, yolov3-tiny_opencv, yolov3_tf, traffic_tf, YOURCUSTOMMODEL_tf...

yolov3_darknet_to_tensorflow(paths=paths, params=params, s3_credentials=s3_credentials)