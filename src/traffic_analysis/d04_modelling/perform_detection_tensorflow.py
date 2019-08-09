from __future__ import division, print_function

import tensorflow as tf
import cv2
import os
import numpy as np

from traffic_analysis.d04_modelling.transfer_learning.tensorflow_detection_utils import read_class_names, \
    remove_overlapping_boxes, letterbox_resize
from traffic_analysis.d04_modelling.transfer_learning.convert_darknet_to_tensorflow import parse_anchors, \
    yolov3_darknet_to_tensorflow
from traffic_analysis.d04_modelling.transfer_learning.generate_tensorflow_model import YoloV3
from traffic_analysis.d04_modelling.perform_detection_opencv import label_detections, \
    choose_objects_of_selected_labels


def perform_detections_in_single_image(image_capture, params, paths, s3_credentials, selected_labels=None):
    """ unifying function (for testing only) w/ tensorflow implementation of yolo to detect objects in a frame
    contains initialization step in each image that is passed through model, which reduces throughput
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            s3_credentials :
            selected_labels :

        Returns:
            bboxes(list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    sess = tf.Session()
    model_initializer, init_data, detection_model = initialize_tensorflow_model(params=params,
                                                                                paths=paths,
                                                                                s3_credentials=s3_credentials,
                                                                                sess=sess)
    boxes, labels, confs = detect_objects_in_image(image_capture=image_capture,
                                                   paths=paths,
                                                   detection_model=detection_model,
                                                   model_initializer=model_initializer,
                                                   init_data=init_data,
                                                   sess=sess,
                                                   selected_labels=selected_labels)
    sess.close()

    return boxes, labels, confs


def initialize_tensorflow_model(params, paths, s3_credentials, sess):
    """ uses pre-existing tensorflow ckpt (or creates one, if it does not yet exist) to initialize variables before
    passing images through neural net for detection
        Args:
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            s3_credentials :
            sess : tensorflow session (pre-load with sess = tf.Session())

        Returns:
            model_initializer(list(list(float))): list of initialized variables bboxes, confs, labels
            init_data (tf.placeholder): initialized array of size of image to be passed through
            detection_model (str): detection model used in the neural network
    """

    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['local_detection_model'], detection_model, 'yolov3.ckpt')

    if not os.path.exists(local_filepath_model):
        yolov3_darknet_to_tensorflow(params=params,
                                     paths=paths,
                                     s3_credentials=s3_credentials)

    conf_thresh = params['detection_confidence_threshold']
    iou_thresh = params['detection_iou_threshold']
    detection_model = params['detection_model']
    local_filepath_model = os.path.join(paths['local_detection_model'], detection_model, 'yolov3.ckpt')

    anchors = parse_anchors(paths)
    class_name_path = os.path.join(paths['local_detection_model'], 'yolov3', 'coco.names')
    classes = read_class_names(class_name_path)
    n_classes = len(classes)

    init_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='init_data')
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


def detect_objects_in_image(image_capture, paths, detection_model, model_initializer, init_data, sess,
                            selected_labels=None):
    """ uses a tensorflow implementation of yolo to detect objects in a frame
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            detection_model (str): detection model used in the neural network
            model_initializer(list(list(float))): list of initialized variables bboxes, confs, labels
            init_data (tf.placeholder): initialized array of size of image to be passed through
            sess : tensorflow session (pre-load with sess = tf.Session())
            selected_labels :
        Returns:
            boxes(list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    image_array, formatting_params = format_image_for_yolo(image_capture)
    boxes_unscaled, confs, label_idxs = sess.run(model_initializer, feed_dict={init_data: image_array})

    # rescale the coordinates to the original image
    boxes = rescale_boxes(boxes_unscaled, formatting_params)
    confs = confs.tolist()

    labels = label_detections(label_idxs=label_idxs,
                              model_name=detection_model,
                              paths=paths)

    if selected_labels is not None:
        boxes, labels, confs = choose_objects_of_selected_labels(bboxes_in=boxes,
                                                                 labels_in=labels,
                                                                 confs_in=confs,
                                                                 selected_labels=selected_labels)

    return boxes, labels, confs


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
    boxes_resized = [[int(np.round(i)) for i in nested] for nested in boxes]

    return boxes_resized
