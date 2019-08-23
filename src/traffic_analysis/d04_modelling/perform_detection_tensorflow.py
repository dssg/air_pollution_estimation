from __future__ import division, print_function
import os

import tensorflow as tf
import cv2
import numpy as np

from traffic_analysis.d04_modelling.transfer_learning.tensorflow_detection_utils import read_class_names, \
    remove_overlapping_boxes, letterbox_resize
from traffic_analysis.d04_modelling.transfer_learning.convert_darknet_to_tensorflow import parse_anchors, \
    yolov3_darknet_to_tensorflow
from traffic_analysis.d04_modelling.transfer_learning.tensorflow_model_loader import YoloV3
from traffic_analysis.d04_modelling.perform_detection_opencv import label_detections, \
    choose_objects_of_selected_labels


def initialize_tensorflow_model(params: dict, 
                                paths: dict, 
                                s3_credentials: dict, 
                                sess: tf.Session) -> (list, tf.placeholder, str):
    """Uses pre-existing tensorflow ckpt (or creates one, if it does not yet exist) to initialize variables before
    passing images through neural net for detection

    Args:
        params: dictionary of parameters from yml file
        paths: dictionary of paths from yml file
        s3_credentials: dictionary of s3 creds from yml file
        sess : tensorflow session (pre-load with sess = tf.Session())

    Returns:
        model_initializer(list(list(float))): list of initialized variables bboxes, confs, labels
        init_data: initialized array of size of image to be passed through
        detection_model: detection model used in the neural network
    """

    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['local_detection_model'], 
                                        detection_model, 
                                        'yolov3.ckpt')

    if not os.path.exists(local_filepath_model):
        yolov3_darknet_to_tensorflow(params=params,
                                     paths=paths,
                                     s3_credentials=s3_credentials)

    conf_thresh = params['detection_confidence_threshold']
    iou_thresh = params['detection_iou_threshold']
    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['local_detection_model'], 
                                        detection_model, 
                                        'yolov3.ckpt')

    anchors = parse_anchors(paths)
    class_name_path = os.path.join(paths['local_detection_model'], 
                                   'yolov3', 
                                   'coco.names')
    classes = read_class_names(class_name_path)
    n_classes = len(classes)

    init_data = tf.placeholder(tf.float32, 
                               [None, 416, 416, 3], 
                               name='init_data')
    yolo_model = YoloV3(n_classes, anchors)
    with tf.variable_scope('YoloV3'):
        feature_map = yolo_model.forward(init_data, False)

    pred_boxes, pred_scores, pred_probs = yolo_model.predict(feature_map)
    pred_confs = pred_scores * pred_probs

    boxes_init, confs_init, labels_init = remove_overlapping_boxes(pred_boxes, pred_confs, n_classes,
                                                                   max_boxes=200, score_thresh=conf_thresh,
                                                                   nms_thresh=iou_thresh)
    model_initializer = [boxes_init, confs_init, labels_init]
    saver = tf.train.Saver()
    saver.restore(sess, local_filepath_model)

    return model_initializer, init_data, detection_model


def detect_objects_tf(images: np.ndarray, 
                      paths: dict, 
                      detection_model: str, 
                      model_initializer: list, 
                      init_data: tf.placeholder, 
                      sess: tf.Session,
                      selected_labels=None) -> (list, list, list):
    """Uses a tensorflow implementation of yolo to detect objects in a frame
    Args:
        image_capture: numpy array containing the captured image (width, height, rbg)
        paths: dictionary of paths from yml file
        detection_model: detection model used in the neural network
        model_initializer(list(list(float))): list of initialized variables bboxes, confs, labels
        init_data: initialized array of size of image to be passed through
        sess: tensorflow session (pre-load with sess = tf.Session())
        selected_labels: labels to return 
    Returns:
        boxes(list(list(int))): width, height, and bottom-left coordinates of detection bboxes
        labels (list(str)): detection labels
        confs (list(float)): detection scores
    """
    formatted_images = []
    formatting_params = []

    for image in images:
        image_array, params = format_image_for_yolo(image)
        formatted_images.append(image_array)
        formatting_params.append(params)

    boxes_unscaled, confs, label_idxs = sess.run(model_initializer, feed_dict={
                                                 init_data: np.squeeze(np.array(formatted_images))
                                                 })

    all_boxes = []
    all_labels = []
    all_confs = []

    for boxes, params, con, labels in zip(boxes_unscaled, formatting_params, confs, label_idxs):
        # rescale the coordinates to the original image
        boxes = np.expand_dims(boxes, axis=0)
        labels = np.expand_dims(labels, axis=0)
        con = np.expand_dims(con, axis=0)
        boxes = reformat_boxes(boxes, params)
        con = con.tolist()

        labels = label_detections(label_idxs=labels,
                                  model_name=detection_model,
                                  paths=paths)

        if selected_labels is not None:
            boxes, labels, con = choose_objects_of_selected_labels(bboxes_in=boxes,
                                                                   labels_in=labels,
                                                                   confs_in=con,
                                                                   selected_labels=selected_labels)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_confs.append(confs)

    return all_boxes, all_labels, all_confs


def format_image_for_yolo(image_capture: np.ndarray) -> (np.ndarray, dict):
    """Formats image capture so it can be ingested by yolov3 model
    Args:
        image_capture: numpy array containing the captured image (width, height, rbg)
    Returns:
        image_capture_formatted: numpy array containing the formatted captured image (width, height, rbg)
        formatting_params: dictionary of parameters returned by letterbox_resize function
    """

    image_capture_resized, resize_ratio, dw, dh = letterbox_resize(
        image_capture, 416, 416)
    image_capture_rgb = cv2.cvtColor(image_capture_resized, cv2.COLOR_BGR2RGB)
    image_capture_rgb_np = np.asarray(image_capture_rgb, np.float32)
    image_capture_formatted = image_capture_rgb_np[np.newaxis, :] / 255.

    formatting_params = {'resize_ratio': resize_ratio,
                         'dw': dw,
                         'dh': dh}

    return image_capture_formatted, formatting_params


def reformat_boxes(boxes_opp_coords: list, 
                   formatting_params: dict) -> list:
    """Rescales bounding boxes to original size of the image
    Args:
        boxes_opp_coords(list(list(int))): list of bottom-left and top-right coordinates of detection boxes
        formatting_params(dict): dictionary of parameters returned by letterbox_resize function
    Returns:
        boxes_resized(list(list(int))): list of bottom-left, width, height coordinates of detection boxes
    """
    dw = formatting_params['dw']
    dh = formatting_params['dh']
    resize_ratio = formatting_params['resize_ratio']

    boxes_opp_coords[:, [0, 2]] = (
        boxes_opp_coords[:, [0, 2]] - dw) / resize_ratio
    boxes_opp_coords[:, [1, 3]] = (
        boxes_opp_coords[:, [1, 3]] - dh) / resize_ratio

    number_of_boxes = len(boxes_opp_coords)
    boxes_width = boxes_opp_coords[:, 2] - boxes_opp_coords[:, 0]
    boxes_height = boxes_opp_coords[:, 3] - boxes_opp_coords[:, 1]

    boxes_reformatted = [[boxes_opp_coords[i][0], boxes_opp_coords[i][1], boxes_width[i], boxes_height[i]]
                         for i in range(number_of_boxes)]

    boxes_reformatted = [[int(i) for i in nested]
                         for nested in boxes_reformatted]

    return boxes_reformatted
