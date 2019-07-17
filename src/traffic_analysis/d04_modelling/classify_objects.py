import numpy as np
from cvlib.object_detection import draw_bbox
import cvlib as cv
import imageio


def classify_objects(videos, names, params, paths, vid_time_length=10, make_videos=True):
    """ this function classifies objects from local mp4 with cvlib python package.
        Args:
            videos (list(nparray)): list of numpy arrays containing the videos
            names (list(str)): list of video names corresponding to the videos
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            vid_time_length (int): length of the video data in seconds
            make_videos (bool): output videos with object classification labels in processed directory
        Returns:
            yolo_dict (dict): nested dictionary where each video is a key for a dict containing:
                obj_bounds (list of np arrays): n-dim list of list of arrays marking the corners of the bounding boxes of objects, for n frames
                obj_labels (list of str): n-dim list of list of labels assigned to classified objects, for n frames
                obj_label_confidences (list of floats): n-dim list of list of floats denoting yolo confidences, for n frames
    """
    yolo_dict = {}

    for video, name in zip(videos, names):
        yolo_dict[name] = {}
        obj_bounds, obj_labels, obj_label_confidences = classify_objects_in_video(
            video, name, params, paths, vid_time_length, make_videos)
        yolo_dict[name]['bounds'] = obj_bounds
        yolo_dict[name]['labels'] = obj_labels
        yolo_dict[name]['confidences'] = obj_label_confidences
    return yolo_dict


def classify_objects_in_video(video, name, params, paths, vid_time_length=10, make_videos=True):
    """ this function classifies objects from local mp4 with cvlib python package.
        Args:
            video (nparray): list of numpy arrays containing the videos
            name (str): list of video names corresponding to the videos
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            vid_time_length (int): length of the video data in seconds
            make_videos (bool): output videos with object classification labels in processed directory

        Returns:
            obj_bounds (list of np arrays): n-dim list of list of arrays marking the corners of the bounding boxes of objects, for n frames
            obj_labels (list of str): n-dim list of list of labels assigned to classified objects, for n frames
            obj_label_confidences (list of floats): n-dim list of list of floats denoting yolo confidences, for n frames
    """
    # loop over frames of video and store in lists
    obj_bounds = []
    obj_labels = []
    obj_label_confidences = []
    cap_cvlib = []

    for i in range(video.shape[0]):
        frame = video[i, :, :, :]

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(
            frame, confidence=params['confidence_threshold'],
            model=params['yolo_model'])
        obj_bounds.append(bbox)
        obj_labels.append(label)
        obj_label_confidences.append(conf)

        # draw bounding box over detected objects
        if make_videos:
            img_cvlib = draw_bbox(frame, bbox, label, conf)
            cap_cvlib.append(img_cvlib)

    # write video to local file
    if make_videos:
        cap_cvlib_npy = np.asarray(cap_cvlib)
        local_mp4_path_out = paths['processed_video'] + name
        imageio.mimwrite(local_mp4_path_out, cap_cvlib_npy,
                         fps=int(video.shape[0] / vid_time_length))

    return obj_bounds, obj_labels, obj_label_confidences
