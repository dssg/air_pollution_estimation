import urllib
import os 
import re
import shutil

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir


def upload_yolo_weights_to_s3(s3_credentials,
							  bucket_name,
							  local_dir,
							  target_dir_on_s3,
							  ):

	delete_and_recreate_dir(temp_dir=local_dir)
######### YOLOV3-TINY
	# get coco.names file
	urllib.request.urlretrieve ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", 
								local_dir + "yolov3-tiny/coco.names")

	# get yolov3-tiny.cfg file
	urllib.request.urlretrieve ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", 
								local_dir + "yolov3-tiny/yolov3-tiny.cfg")

	# get yolov3-tiny.weights file
	urllib.request.urlretrieve ("https://pjreddie.com/media/files/yolov3-tiny.weights", 
								local_dir + "yolov3-tiny/yolov3-tiny.weights")

######### YOLOV3
	# get coco.names file
	urllib.request.urlretrieve ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", 
								local_dir + "yolov3/coco.names")

    # get yolov3.weights
	urllib.request.urlretrieve ("https://pjreddie.com/media/files/yolov3.weights", 
								local_dir + "yolov3/yolov3.weights")
	# get yolov3.cfg
	urllib.request.urlretrieve ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", 
								local_dir + "yolov3/yolov3.cfg")

	# get yolov3_anchors.txt
	urllib.request.urlretrieve ("https://raw.githubusercontent.com/wizyoung/YOLOv3_TensorFlow/master/data/yolo_anchors.txt", 
								local_dir + "yolov3/yolo_anchors.txt")


############# TENSORFLOW???
	# TODO: GET TENSORFLOW WEIGHTS FROM STORAGE 

############# UPLOAD TO FOLDER
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

	# Set the directory you want to start from
	for dir_path, sub_dir_list, file_list in os.walk(local_dir):
	    print('Found directory: %s' % dir_path)
	    dir_name = re.split(r'\\|/', dir_path)[-1]
	    for file_name in file_list:
	    	path_of_file_to_upload = os.path.join(dir_name, file_name)
	    	path_to_upload_file_to = target_dir_on_s3 + dir_name +"/" + file_name

	    	print(f"uploading file {path_of_file_to_upload} to {path_to_upload_file_to}")
	    	dl.upload_file(path_of_file_to_upload=path_of_file_to_upload, 
    		               path_to_upload_file_to=path_to_upload_file_to)

	shutil.rmtree(local_dir)


def upload_camera_details_to_s3(s3_credentials,
                                bucket_name,
                                local_dir,
                                target_dir_on_s3
                                ):

	camera_details_path = local_dir + "camera_details.json"
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])
    dl.upload_file(path_of_file_to_upload=camera_details_path, 
	               path_to_upload_file_to=target_dir_on_s3)

	
