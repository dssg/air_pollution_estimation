from src.d00_utils.bbox_helpers import color_bboxes, bboxcvlib_to_bboxcv2
from src.d00_utils.load_confs import load_parameters
import numpy as np
import cvlib
import os
import yaml

params = load_parameters()


def detect_bboxes(frame: np.ndarray, model: str,
                  implementation: str = None, selected_labels=True) -> (list, list, list):
    '''Detect bounding boxes on a frame using specified model and optionally an implementation.
    bboxes returned in format (xmin, ymin, w, h). Colors are assigned to bboxes by the type. 

    Keyword arguments 

    frame -- one frame of a video 
    model -- specify the name of an object model to use 
    implementation -- specify the implementation of the model to use 
    selected_labels -- if a list of labels is supplied, only bboxes with these labels will be returned
    '''
    if implementation == 'cvlib':
        if model == 'yolov3-tiny':
            bboxes_cvlib, labels, confs = cvlib.detect_common_objects(frame, confidence=params['detection_confidence_threshold'],
                                                                      nms_thresh=params['detection_nms_threshold'], model='yolov3-tiny')
            bboxes_cv2 = [bboxcvlib_to_bboxcv2(
                bbox_cvlib) for bbox_cvlib in bboxes_cvlib]
    # sample architecture for how other models/implementations could be added
    elif implementation == 'jacks_yolo_implementation':
        pass
    # if a list of selected_labels has been specified, remove bboxes, labels, confs which
    # do not correspond to labels in selected_labels
    del_inds = []
    if selected_labels:
        for i, label in enumerate(labels):
            # specify object types to ignore
            if label not in params["selected_labels"]:
                del_inds.append(i)

        # delete items from lists in reverse to avoid index shifting issue
        for i in sorted(del_inds, reverse=True):
            del bboxes_cv2[i]
            del labels[i]
            del confs[i]

    return bboxes_cv2, labels, confs
