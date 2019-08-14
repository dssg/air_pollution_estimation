import time
import cv2
import sys
import os
import random

ospath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(ospath)

from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d04_modelling.perform_detection_tensorflow import perform_detections_in_single_image as tensorflow_detect
from traffic_analysis.d04_modelling.perform_detection_opencv import detect_objects_in_image as opencv_detect


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

    elif detection_method == 'yolov3_traffic':
        start_time = time.time()
        bbox, label, confidence = tensorflow_detect(imcap, paths=paths, params=params, s3_credentials=s3_credentials,
                                                    selected_labels=None)
        end_time = time.time()

    for box in bbox:
        color = [random.randint(0, 255) for _ in range(3)]
        c1 = (box[0], box[1])
        c2 = (box[2]+box[0], box[1]+box[3])
        cv2.rectangle(imcap, c1, c2, color)

    cv2.imwrite(image_path[:-4] + 'out' + image_path[-4:], imcap)

    print(label)
    print(bbox)
    print(imcap.shape)


# test_detection('C:/Users/joh3146/Documents/dssg/air_pollution_estimation/data/frame_level/frame001.jpg')
test_detection('/home/jack_hensley/air_pollution_estimation/data/frame_level/frame001.jpg')
