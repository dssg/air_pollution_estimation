import time
import cv2
import sys
print(sys.path)

from src.traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from src.traffic_analysis.d04_modelling.perform_detection_tensorflow import detect_objects_in_image as tensorflow_detect
from src.traffic_analysis.d04_modelling.perform_detection_opencv import detect_objects_in_image as opencv_detect


def test_detection(image_path):
    paths = load_paths()
    params = load_parameters()
    s3_credentials = load_credentials()
    detection_method = params['detection_model']

    imcap = cv2.imread(image_path)

    if detection_method == 'yolov3':
        start_time = time.time()
        bbox, label, confidence = opencv_detect(imcap, paths=paths, params=params, s3_credentials=s3_credentials)
        end_time = time.time()

    elif detection_method == 'yolov3-tiny':
        start_time = time.time()
        bbox, label, confidence = opencv_detect(imcap, paths=paths, params=params, s3_credentials=s3_credentials)
        end_time = time.time()

    elif detection_method == 'yolov3_tf':
        start_time = time.time()
        bbox, label, confidence = tensorflow_detect(imcap, paths=paths, params=params, s3_credentials=s3_credentials)
        end_time = time.time()

    delt_time = end_time - start_time
    print(delt_time)
    print(bbox, label, confidence)


test_detection('/home/jack_hensley/air_pollution_estimation/data/frame_level/frame001.jpg')
