# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function
import os
import tensorflow as tf

from traffic_analysis.d00_utils.tf_model import yolov3
from traffic_analysis.d00_utils.tf_yolo_weights_utils import parse_anchors, load_weights


def darknet_to_tensorflow(paths, params):
    """ builds yolov3 in a tensorflow model from darknet
        Args:
            paths:
            params:
    """

    model_file_path = paths['detect_model']
    detection_model = params['detection_model']

    if not detection_model == 'yolov3':  # can only use yolov3, not yolov3-tiny as of now
        pass

    else:
        num_class = 80
        img_size = 416
        weight_path = os.path.join(model_file_path, 'yolov3_darknet', 'yolov3.weights')
        save_path = os.path.join(model_file_path, 'yolov3_tf', 'yolov3.ckpt')
        anchors = parse_anchors()

        model = yolov3(num_class, anchors)

        with tf.Session() as sess:
            inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

            with tf.variable_scope('yolov3'):
                feature_map = model.forward(inputs)

            saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

            load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
            sess.run(load_ops)
            saver.save(sess, save_path=save_path)
