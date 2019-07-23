from traffic_analysis.d00_utils.bbox_helpers import bboxcvlib_to_bboxcv2
from traffic_analysis.d00_utils.load_confs import load_parameters
import numpy as np
import cvlib

def detect_bboxes(frame: np.ndarray, model: str,
                  detection_confidence_threshold: float,
                  detection_nms_threshold: float, 
                  implementation: str = None, 
                  selected_labels: str = None) -> (list, list, list):
    '''Detect bounding boxes on a frame using specified model and optionally an implementation.
    bboxes returned in format (xmin, ymin, w, h). Colors are assigned to bboxes by the type. 

    Keyword arguments 

    frame -- one frame of a video 
    model -- specify the name of an object model to use
    implementation -- specify the implementation of the model to use 
    selected_labels -- if a list of labels is supplied, only bboxes with these labels will be returned
    detection_confidence_threshold -- conf above which to return label 
    detection_nms_threshold -- yolo param
    '''
    if implementation == 'cvlib':
        if model == 'yolov3-tiny':
            bboxes_cvlib, labels, confs = cvlib.detect_common_objects(frame, 
                                          confidence=detection_confidence_threshold,
                                          nms_thresh=detection_nms_threshold, 
                                          model='yolov3-tiny')
            bboxes_cv2 = [bboxcvlib_to_bboxcv2(bbox_cvlib) for bbox_cvlib in bboxes_cvlib]
    # sample architecture for how other models/implementations could be added
    elif implementation == 'other_implementation':
        pass
    # if a list of selected_labels has been specified, remove bboxes, labels, confs which
    # do not correspond to labels in selected_labels
    del_inds = []
    if selected_labels is not None:
        for i, label in enumerate(selected_labels):
            # specify object types to ignore
            if label not in :
                del_inds.append(i)

        # delete items from lists in reverse to avoid index shifting issue
        for i in sorted(del_inds, reverse=True):
            del bboxes_cv2[i]
            del labels[i]
            del confs[i]

    return bboxes_cv2, labels, confs
