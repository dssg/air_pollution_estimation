import os

import numpy as np
import cv2

from traffic_analysis.d02_ref.download_detection_model_from_s3 import download_detection_model_from_s3


def detect_objects_cv(image_capture: np.ndarray,
                      params: dict,
                      paths: dict,
                      s3_credentials: dict,
                      selected_labels: list = None) -> (list, list, list):
    """Unifying function that defines the detected objects in an image
    Args:
        image_capture: numpy array containing the captured image (width, height, rbg)
        params: dictionary of parameters from yml file
        paths: dictionary of paths from yml file
        s3_credentials: s3 credentials
        selected_labels: list of labels if supplied that returns only bboxes with these labels

    Returns:
        bboxes(list(list(int))): list of bottom-left coordinates, width, height of detection bboxes
        labels (list(str)): list of detection labels
        confs (list(float)): list of detection scores
    """

    conf_thresh = params['detection_confidence_threshold']
    detection_iou_threshold = params['detection_iou_threshold']
    model_name = params['detection_model']

    download_detection_model_from_s3(model_name=model_name,
                                     paths=paths,
                                     s3_credentials=s3_credentials)
    network_output = pass_image_through_nn(image_capture=image_capture,
                                           model_name=model_name,
                                           paths=paths)
    boxes_unfiltered, label_idxs_unfiltered, confs_unfiltered = get_detected_objects(image_capture=image_capture,
                                                                                     network_output=network_output,
                                                                                     conf_thresh=conf_thresh)
    boxes, label_idxs, confs = reduce_overlapping_detections(bboxes_in=boxes_unfiltered,
                                                             label_idxs_in=label_idxs_unfiltered,
                                                             confs_in=confs_unfiltered,
                                                             conf_thresh=conf_thresh,
                                                             iou_thresh=detection_iou_threshold)
    labels = label_detections(model_name=model_name,
                              paths=paths,
                              label_idxs=label_idxs)
    if selected_labels is not None:
        boxes, labels, confs = choose_objects_of_selected_labels(bboxes_in=boxes,
                                                                 labels_in=labels,
                                                                 confs_in=confs,
                                                                 selected_labels=selected_labels)

    return boxes, labels, confs


def populate_labels(model_origin: str,
                    paths: dict) -> list:
    """Report full list of object labels corresponding to detection model of choice
    Args:
        model_origin: name of the model folder in which the names file exists (i.e., yolov3_opencv for yolov3_tf)
        paths: dictionary of paths from yml file

    Returns:
        labels (list(str)): list of object labels strings
    """

    model_file_path = paths['local_detection_model']
    labels_file_path = os.path.join(model_file_path, model_origin, 'coco.names')
    f = open(labels_file_path, 'r')
    labels = [line.strip() for line in f.readlines()]

    return labels


def make_bbox_around_object(image_capture: np.ndarray,
                            unscaled_bbox: np.ndarray) -> list:
    """Makes bounding boxes around detected objects at original scale of image
    Args:
        unscaled_bbox: nparray with unscaled width, height, and bottom-left coordinates of detected object
        image_capture: nparray containing the captured image (width, height, rbg)

    Returns:
        scaled_bbox (list(int)): bottom-left coordinates, width, height of bounding box
    """

    image_capture_height, image_capture_width = image_capture.shape[:2]
    center_x = int(unscaled_bbox[0] * image_capture_width)
    center_y = int(unscaled_bbox[1] * image_capture_height)
    w = int(unscaled_bbox[2] * image_capture_width)
    h = int(unscaled_bbox[3] * image_capture_height)
    x = center_x - w / 2
    y = center_y - h / 2
    scaled_bbox = [x, y, w, h]

    return scaled_bbox


def identify_most_probable_object(grid_cell_estimate: np.ndarray) -> (int, float):
    """Finds the most likely object to exist in a specific grid cell of image
    Args:
        grid_cell_estimate: nparray with scores of object labels in grid cell

    Returns:
        most_probable_object_idx: index of label of most probable object
        most_probable_object_score: score (i.e., confidence) of most probable object detected in image
    """

    scores = grid_cell_estimate[5:]  # ignore the physical parameters
    most_probable_object_idx = np.argmax(scores)
    most_probable_object_score = scores[most_probable_object_idx]

    return most_probable_object_idx, most_probable_object_score


def pass_image_through_nn(image_capture: np.ndarray,
                          model_name: str,
                          paths: dict) -> list:
    """Detection model generates scores (i.e., confidence) of each object existing in image
    Args:
        image_capture: numpy array containing the captured image (width, height, rbg)
        model_name: name of the model to use
        paths: dictionary of paths from yml file

    Returns:
        output_layers (list(nparray)): list of neural network output layers and scores of predicted objects
    """
    # pre-process image:
    # scaling
    # Turn into the right shape for the NN (here 3x416x416)
    # and align the BGR channel order of open cv with the RGB order of mean values
    scale = 0.00392  # required scaling for yolo
    pre_processed_image = cv2.dnn.blobFromImage(image=image_capture,
                                                scalefactor=scale,
                                                # spatial size expected by CNN
                                                size=(416, 416),
                                                # do not use mean subtraction
                                                mean=(0, 0, 0),
                                                swapRB=True,
                                                crop=False)

    # read model as deep neural network in opencv
    config_file_path = os.path.join(
        paths['local_detection_model'], model_name, model_name + '.cfg')
    weights_file_path = os.path.join(
        paths['local_detection_model'], model_name, model_name + '.weights')
    # can use other net, see documentation
    net = cv2.dnn.readNetFromDarknet(config_file_path, weights_file_path)

    # input image to neural network
    net.setInput(pre_processed_image)

    # forward pass of blob through neural network
    layer_names = net.getLayerNames()
    ids_of_output_layers = [i[0] for i in net.getUnconnectedOutLayers()]
    # need to offset by 1 since Python starts counting at 0
    names_of_output_layers = [layer_names[i - 1] for i in ids_of_output_layers]

    network_output = net.forward(names_of_output_layers)

    return network_output


def get_detected_objects(image_capture: np.ndarray,
                         network_output: list,
                         conf_thresh: float) -> (list, list, list):
    """Describes the detections that score above the confidence threshold
    Args:
        image_capture: numpy array containing the captured image (width, height, rbg)
        network_output (list(nparray)): list of neural network outputs and scores of predicted objects
        conf_thresh: minimum confidence required in object detection, between 0 and 1
    Returns:
        bboxes (list(list(int))): list of width, height, and bottom-left coordinates of detection bounding boxes
        label_idxs (list(int)): list of indices corresponding to the detection labels
        confs (list(float)): list of scores of detections
    """
    bboxes = []
    label_idxs = []
    confs = []

    for output_layer in network_output:  # loop through outputs from the different output layers
        for grid_cell_estimates in output_layer:  # loop through grid cells in output layer

            # find most likely object in specific grid cell of image
            object_label_idx, max_conf = identify_most_probable_object(
                grid_cell_estimate=grid_cell_estimates)

            # append object to running list of objects if prediction score is above confidence threshold
            if max_conf > conf_thresh:
                object_bbox = make_bbox_around_object(image_capture=image_capture,
                                                      unscaled_bbox=grid_cell_estimates)
                bboxes.append(object_bbox)
                label_idxs.append(object_label_idx)
                confs.append(float(max_conf))

    return bboxes, label_idxs, confs


def reduce_overlapping_detections(bboxes_in: list,
                                  label_idxs_in: list,
                                  confs_in: list,
                                  conf_thresh: float,
                                  iou_thresh: float) -> (list, list, list):
    """ Femoves the detections that score above the nms threshold
    That is:
    1) Discard detections with probability to be present below conf threshold
    2) For each cell, keep the detection with the highest confidence
    3) Discard predictions with iou above the iou threshold

    Args:
        bboxes_in (list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
        label_idxs_in (list(int)): list of indices corresponding to the detection labels
        confs_in (list(float)): list of scores of detections
        conf_thresh: minimum confidence required in object detection, between 0 and 1
        iou_thresh: non maximum suppression (nms) threshold to select for maximum overlap allowed between bboxes
    Returns:
        bboxes_out (list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
        label_idxs_out (list(int)): list of indices corresponding to the detection labels
        confs_out (list(float)): list of detection scores
    """
    idx_boxes_nms = cv2.dnn.NMSBoxes(bboxes=bboxes_in,
                                     scores=confs_in,
                                     score_threshold=conf_thresh,
                                     nms_threshold=iou_thresh)

    # initialize output lists
    bboxes_out = []
    label_idxs_out = []
    confs_out = []

    # select "ins" of the reported indices
    for i in idx_boxes_nms:
        i = i[0]
        bbox = bboxes_in[i]
        x, y, w, h = bbox[:4]
        bboxes_out.append([round(x), round(y), round(w), round(h)])
        label_idxs_out.append(label_idxs_in[i])
        confs_out.append(confs_in[i])

    return bboxes_out, label_idxs_out, confs_out


def label_detections(label_idxs: list,
                     model_name: str,
                     paths: dict) -> list:
    """Labels the detected objects according to their index in list
    Args:
        label_idxs (list(int)): list of indices corresponding to the detection labels
        model_name: name of the model to use
        paths (dict): dictionary of paths from yml file

    Returns:
        labels (list(str)): labels of the reported object detections
    """

    # import the list of labels
    if model_name[-2:] == 'tf':
        model_origin = 'yolov3'
    else:
        model_origin = model_name.split('_')[0]
    label_list = populate_labels(model_origin=model_origin,
                                 paths=paths)

    # initialize the output list
    labels = []

    # select label names according to indices selected
    for tick in label_idxs:
        labels.append(str(label_list[tick]))

    return labels


def choose_objects_of_selected_labels(bboxes_in: list,
                                      labels_in: list,
                                      confs_in: list,
                                      selected_labels: list) -> (list, list, list):
    """Removes detections that correspond to labels outside of selected ones, if specified
    Args:
        bboxes_in (list(list(int))): width, height, and bottom-left coordinates of detection bboxes
        labels_in (list(int)): indices corresponding to the detection labels
        confs_in (list(float)): detection scores
        selected_labels (list(str)): labels; if supplied will only returns bboxes with these labels
    Returns:
        bboxes_out (list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
        label_idxs_out (list(int)): indices corresponding to the detection labels
        confs_out (list(float)): detection scores
    """

    del_idxs = []
    if selected_labels is not None:
        for i, detected_label in enumerate(labels_in):
            # specify object types to ignore
            if detected_label not in selected_labels:
                del_idxs.append(i)

        bboxes_out = bboxes_in
        labels_out = labels_in
        confs_out = confs_in
        # delete items from lists in reverse to avoid index shifting issue
        for i in sorted(del_idxs, reverse=True):
            del bboxes_out[i]
            del labels_out[i]
            del confs_out[i]

    return bboxes_out, labels_out, confs_out
