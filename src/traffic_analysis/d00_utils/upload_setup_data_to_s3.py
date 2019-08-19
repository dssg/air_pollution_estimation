import urllib
import os 
import re
import shutil
import glob

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name


def upload_yolo_weights_to_s3(s3_credentials,
                              bucket_name,
                              local_dir,
                              target_dir_on_s3,
                              ):

    delete_and_recreate_dir(temp_dir=local_dir)

    download_dict = {os.path.join(local_dir, "yolov3-tiny"): ["https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                                                              "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                                                               "https://pjreddie.com/media/files/yolov3-tiny.weights"], 
                    os.path.join(local_dir, "yolov3"): ["https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                                                        "https://pjreddie.com/media/files/yolov3.weights",
                                                        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                                                        "https://raw.githubusercontent.com/wizyoung/YOLOv3_TensorFlow/master/data/yolo_anchors.txt"
                                                        ]
                    }

    for download_dir, download_urls in download_dict.items(): 
        os.makedirs(download_dir)
        for download_url in download_urls: 
            filename = download_url.split("/")[-1]
            download_path = os.path.join(download_dir, filename) 

            try: 
                urllib.request.urlretrieve(download_url, 
                                           download_path)
                print(f"Successfully downloaded {download_url}")

            except Exception as e: 
                print(e)
                print("Failed to download url ", download_url)

    ############# TENSORFLOW???
    # TODO: GET TENSORFLOW WEIGHTS FROM STORAGE 

    ############# UPLOAD TO S3 bucket
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=bucket_name)

    # Set the directory you want to start from
    for dir_path, sub_dir_list, file_list in os.walk(local_dir):
        print('Found directory: %s' % dir_path)
        dir_name = re.split(r'\\|/', dir_path)[-1]
        if dir_name == "setup": 
            continue

        for file_name in file_list:
            path_of_file_to_upload = os.path.join(dir_path, file_name)
            path_to_upload_file_to = target_dir_on_s3 + dir_name +"/" + file_name

            print(f"uploading file {path_of_file_to_upload} to {path_to_upload_file_to}")
            dl.upload_file(path_of_file_to_upload=path_of_file_to_upload, 
                           path_to_upload_file_to=path_to_upload_file_to)

    shutil.rmtree(local_dir)

def upload_annotations_to_s3(s3_credentials, paths):
    
    data_loader = DataLoaderS3(s3_credentials=s3_credentials, bucket_name=paths["bucket_name"])

    # raw videos
    for video_file in glob.glob(paths["setup_video"] + "*.mp4"):
        camera_id, video_upload_datetime = parse_video_or_annotation_name(video_name=video_file.split('/')[-1])
        data_loader.upload_file(path_of_file_to_upload=video_file, path_to_upload_file_to=paths["s3_video"] + "date_test/" + str(video_upload_datetime) + "/" + video_file.split('/')[-1])

    # xml files
    for xml_file in glob.glob(paths["setup_xml"] + "*.xml"):
        data_loader.upload_file(path_of_file_to_upload=xml_file, path_to_upload_file_to=paths["s3_annotations"] + "cvat_test/" + xml_file.split('/')[-1])

    return
