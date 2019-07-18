import numpy as np
from cvlib.object_detection import draw_bbox
import cvlib as cv
import imageio

from traffic_analysis.d05_reporting.report_yolo import yolo_output_df


def classify_objects(video_dict, params, paths, vid_time_length=10, make_videos=True):
    """ this function classifies objects from video dict with cvlib python package.
        Args:
            video_dict (dict): dict of numpy arrays containing all the jamcam videos
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            vid_time_length (int): length of the video data in seconds
            make_videos (bool): output videos with object classification labels in processed directory
        Returns:
            frame_level_df (df): pandas dataframe containing the results of Yolo object detection
    """
    yolo_dict = {}

    for video_num, (name, video) in enumerate(video_dict.items()):

        print('Classifying video {}/{}.'.format(video_num, len(video_dict)))
        yolo_dict[name] = {}

        # loop over frames of video and store in lists
        obj_bounds = []
        obj_labels = []
        obj_label_confidences = []
        cap_cvlib = []

        for i in range(video.shape[0]):
            frame = video[i, :, :, :]

            # apply object detection
            bbox, label, conf = cv.detect_common_objects(frame, confidence=params['detection_confidence_threshold'],
                                                         model=params['detection_model'])
            obj_bounds.append(bbox)
            obj_labels.append([l.replace('motorcycle', 'motorbike') for l in label])
            obj_label_confidences.append(conf)

            # draw bounding box over detected objects
            if make_videos:
                img_cvlib = draw_bbox(frame, bbox, label, conf)
                cap_cvlib.append(img_cvlib)

        # write video to local file
        if make_videos:
            cap_cvlib_npy = np.asarray(cap_cvlib)
            local_mp4_path_out = paths['processed_video'] + name
            imageio.mimwrite(local_mp4_path_out, cap_cvlib_npy, fps=int(video.shape[0] / vid_time_length))

        yolo_dict[name]['bounds'] = obj_bounds
        yolo_dict[name]['labels'] = obj_labels
        yolo_dict[name]['confidences'] = obj_label_confidences

    frame_level_df = yolo_output_df(yolo_dict)

    return frame_level_df
