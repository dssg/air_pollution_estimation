# requirements
import boto3
import cv2
import numpy as np
import os
from cvlib.object_detection import draw_bbox
import cvlib as cv
import imageio
imageio.plugins.ffmpeg.download()
import time


def s3_to_local_mp4(camera, date, time, extension, local_mp4_path):
    """ download mp4 to working directory """

    # create s3 file path based on desired camera
    timestamp = date[:4] + "-" + date[4:6] + "-" + date[6:] + "_" + time[:2] + '.' + time[
                                                                                     2:]  # assumes datetime = date as yyyymmdd + time as hhmm
    s3_vid_key = "raw" + "/" + "video_data" + "/" + camera + "/" + timestamp + extension

    # convert to s3 bucket
    s3_session = boto3.Session(profile_name='dssg')
    s3_resource = s3_session.resource('s3')
    bucket_name = 'air-pollution-uk'
    s3_bucket = s3_resource.Bucket(bucket_name)

    # download to working directory of choice
    s3_bucket.download_file(s3_vid_key, local_mp4_path)


def mp4_to_npy(local_mp4_path):
    """ create np array file from mp4 file in same directory """

    # use cv2 to read in mp4 file
    vid_mp4 = cv2.VideoCapture(local_mp4_path)

    # setup vid_array np skeleton according to captured file
    frame_count = int(vid_mp4.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid_mp4.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_mp4.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_array = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    # feed video data into vid_array
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, vid_array[fc] = vid_mp4.read()
        fc += 1

    # not sure what this does
    vid_mp4.release()
    cv2.waitKey(0)

    # save file to mp4 directory as .npy file
    pre, ext = os.path.splitext(local_mp4_path)
    np.save(pre, vid_array)

    return vid_array


def classify_objects(local_mp4_path, confidence_threshold=0.25, vid_time_length=10, make_video=True):
    """ classify objects from local mp4 with cvlib """

    start_time = time.time()

    # import video from local path
    cap = cv2.VideoCapture(local_mp4_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fps = int(n_frames / vid_time_length)  # assumes vid_length in seconds
    yolo_model = 'yolov3-tiny'  # consider changing this for higher accuracy

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
        bbox, label, conf = cv.detect_common_objects(frame, confidence=confidence_threshold,
                                                     model=yolo_model)
        obj_bounds.append(bbox)
        obj_labels.append(label)
        obj_label_confidences.append(conf)

        # draw bounding box over detected objects
        if make_video == True:
            img_cvlib = draw_bbox(frame, bbox, label, conf)
            cap_cvlib.append(img_cvlib)
        else:
            pass

    # write video to local file
    if make_video == True:
        cap_cvlib_npy = np.asarray(cap_cvlib)
        local_mp4_path_out = local_mp4_path[:-4] + '_cvlib' + local_mp4_path[-4:]
        imageio.mimwrite(local_mp4_path_out, cap_cvlib_npy, fps=cap_fps)
    else:
        pass

    print('Run time is %s seconds' % (time.time() - start_time))
    return obj_bounds, obj_labels, obj_label_confidences
