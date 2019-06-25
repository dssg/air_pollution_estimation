import cv2
import numpy as np
from cvlib.object_detection import draw_bbox
import cvlib as cv
import imageio
import time
import os


def classify_objects(local_mp4_path, vid_time_length=10, params, make_video=True):
    """ this function classifies objects from local mp4 with cvlib python package.
        Args:
            local_mp4_path (str): path to mp4 file in directory
            vid_time_length (int): length of the video data in seconds
            make_video (bool): output a video with object classification labels in same directory as original video

        Returns:
            obj_bounds (list of np arrays): n-dim list of list of arrays marking the corners of the bounding boxes of objects, for n frames
            obj_labels (list of str): n-dim list of list of labels assigned to classified objects, for n frames
            obj_label_confidences (list of floats): n-dim list of list of floats denoting yolo confidences, for n frames
    """

    start_time = time.time()

    # import video from local path
    cap = cv2.VideoCapture(local_mp4_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fps = int(n_frames / vid_time_length)  # assumes vid_length in seconds

    # loop over frames of video and store in lists
    obj_bounds = []
    obj_labels = []
    obj_label_confidences = []
    cap_cvlib = []

    while cap.isOpened():
        # open imported video
        status, frame = cap.read()
        if not status:
            break

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame, confidence=params['confidence_threshold'],
                                                     model=params['yolo_model'])
        obj_bounds.append(bbox)
        obj_labels.append(label)
        obj_label_confidences.append(conf)

        # draw bounding box over detected objects
        if make_video:
            img_cvlib = draw_bbox(frame, bbox, label, conf)
            cap_cvlib.append(img_cvlib)
        else:
            pass

    # write video to local file
    if make_video:
        cap_cvlib_npy = np.asarray(cap_cvlib)
        local_mp4_path_out = local_mp4_path[:-4] + '_cvlib' + local_mp4_path[-4:]
        imageio.mimwrite(local_mp4_path_out, cap_cvlib_npy, fps=cap_fps)
    else:
        pass

    #print('Run time is %s seconds' % (time.time() - start_time))
    return obj_bounds, obj_labels, obj_label_confidences
