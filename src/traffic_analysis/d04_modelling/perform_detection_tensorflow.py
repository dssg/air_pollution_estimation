# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import cv2
import os
import numpy as np

from traffic_analysis.d00_utils.tensorflow_detection_utils import read_class_names, remove_overlapping_boxes, \
    letterbox_resize
from traffic_analysis.d00_utils.convert_darknet_to_tensorflow import parse_anchors
from traffic_analysis.d00_utils.generate_tensorflow_model import YoloV3
from traffic_analysis.d00_utils.convert_darknet_to_tensorflow import yolov3_darknet_to_tensorflow


def detect_objects_in_image(image_capture, params, paths):
    """ uses a tensorflow implementation of yolo to detect objects in a frame
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file

        Returns:
            bboxes(list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['detection_model'], detection_model, 'yolov3.ckpt')

    if detection_model == 'yolov3_tf':  # only use with yolov3_tf as detection model
        if not os.path.exists(local_filepath_model):  # create yolov3 tensorflow model on local if does not exist
            yolov3_darknet_to_tensorflow(params, paths)

        boxes, confs, labels = pass_image_through_nn(image_capture, paths, params)

    return boxes, confs, labels


def pass_image_through_nn(image_capture, paths, params):
    """ passes an image through a saved yolov3 neural network
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file

        Returns:
            boxes(list(list(int))): list of width, height, and bottom-left coordinates of detection boxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    conf_thresh = params['detection_confidence_threshold']
    iou_thresh = params['detection_iou_threshold']
    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['detection_model'], detection_model, 'yolov3.ckpt')

    # prepare the inputs for model initialization
    image_array, formatting_params = format_image_for_yolo(image_capture)
    anchors = parse_anchors(paths)
    class_name_path = os.path.join(paths['detection_model'], 'yolov3', 'coco.names')
    classes = read_class_names(class_name_path)
    n_classes = len(classes)

    with tf.Session() as sess:

        # initialize tensorflow yolov3 model
        init_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='init_data')
        yolo_model = YoloV3(n_classes, anchors)
        with tf.variable_scope('yolov3'):
            feature_map = yolo_model.forward(init_data, False)

        pred_boxes, pred_confs, pred_probs = yolo_model.predict(feature_map)
        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = remove_overlapping_boxes(pred_boxes, pred_scores, n_classes,
                                                         max_boxes=200, score_thresh=conf_thresh,
                                                         nms_thresh=iou_thresh)

        saver = tf.train.Saver()
        saver.restore(sess, local_filepath_model)
        boxes_unscaled, scores_out, labels_out = sess.run([boxes, scores, labels], feed_dict={init_data: image_array})

        # rescale the coordinates to the original image
        boxes_out = rescale_boxes(boxes_unscaled, formatting_params)

    return boxes_out, scores_out, labels_out


def format_image_for_yolo(image_capture):
    """ formats image capture so it can be ingested by yolov3 model
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
        Returns:
            image_capture_formatted (nparray): numpy array containing the formatted captured image (width, height, rbg)
            formatting_params(dict): dictionary of parameters returned by letterbox_resize function
    """

    image_capture_resized, resize_ratio, dw, dh = letterbox_resize(image_capture, 416, 416)
    image_capture_rgb = cv2.cvtColor(image_capture_resized, cv2.COLOR_BGR2RGB)
    image_capture_rgb_np = np.asarray(image_capture_rgb, np.float32)
    image_capture_formatted = image_capture_rgb_np[np.newaxis, :] / 255.

    formatting_params = {'resize_ratio': resize_ratio,
                         'dw': dw,
                         'dh': dh}

    return image_capture_formatted, formatting_params


def rescale_boxes(boxes, formatting_params):
    """ rescales bounding boxes to original size of the image
        Args:
            boxes(list(list(int))): list of width, height, and bottom-left coordinates of detection boxes
            formatting_params(dict): dictionary of parameters returned by letterbox_resize function
        Returns:
            boxes_resized(list(list(int))): list of width, height, and bottom-left coordinates of detection boxes
    """
    dw = formatting_params['dw']
    dh = formatting_params['dh']
    resize_ratio = formatting_params['resize_ratio']

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / resize_ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / resize_ratio
    boxes_resized = boxes

    return boxes_resized

