import numpy as np
from cvlib.object_detection import draw_bbox
import cvlib as cv
import imageio


def classify_objects(video, params, paths, vid_time_length=10, make_video=True, vid_name='unnamed'):
    """ this function classifies objects from local mp4 with cvlib python package.
        Args:
            video (nparray): numpy array containing the video
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            vid_time_length (int): length of the video data in seconds
            make_video (bool): output a video with object classification labels in same directory as original video
            vid_name (str): name of the output video

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
        bbox, label, conf = cv.detect_common_objects(frame, confidence=params['confidence_threshold'],
                                                     model=params['yolo_model'])
        obj_bounds.append(bbox)
        obj_labels.append(label)
        obj_label_confidences.append(conf)

        # draw bounding box over detected objects
        if make_video:
            img_cvlib = draw_bbox(frame, bbox, label, conf)
            cap_cvlib.append(img_cvlib)

    # write video to local file
    if make_video:
        cap_cvlib_npy = np.asarray(cap_cvlib)
        local_mp4_path_out = paths['local_video'] + vid_name + '.mp4'
        imageio.mimwrite(local_mp4_path_out, cap_cvlib_npy, fps=int(video.shape[0] / vid_time_length))

    return obj_bounds, obj_labels, obj_label_confidences
