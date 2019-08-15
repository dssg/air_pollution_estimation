# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function
import os
import tensorflow as tf
import numpy as np

from traffic_analysis.d04_modelling.transfer_learning.generate_tensorflow_model import YoloV3
from traffic_analysis.d02_ref.download_detection_model_from_s3 import download_detection_model_from_s3


def yolov3_darknet_to_tensorflow(paths,
                                 params,
                                 s3_credentials):
    """ saves a tensorflow model build of yolo, taken from darknet weights
        Args:
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            s3_credentials (dict): s3 credentials
    """

    model_file_path = paths['local_detection_model']
    detection_model = params['detection_model']

    if not detection_model == 'yolov3_tf':  # can only create tf model with yolov3 darknet weights
        pass

    else:
        if not os.path.exists(os.path.join(model_file_path, 'yolov3')):
            download_detection_model_from_s3(model_name='yolov3',
                                             paths=paths,
                                             s3_credentials=s3_credentials)

        if not os.path.exists(os.path.join(model_file_path, detection_model)):
            os.mkdir(os.path.join(model_file_path, detection_model))

        num_class = 80
        img_size = 416
        weight_path = os.path.join(model_file_path, 'yolov3', 'yolov3.weights')
        save_path = os.path.join(model_file_path, 'yolov3_tf', 'yolov3.ckpt')
        anchors = parse_anchors(paths)

        # build tensorflow model as a yolov3 class
        model = YoloV3(num_class, anchors)

        # save model locally as tensorflow .ckpt
        with tf.Session() as sess:
            #with tf.device('/device:GPU:' + str(params['GPU_number'])):
            # tf model initialization
            inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

            with tf.variable_scope('YoloV3'):  # i think this generates the model output nodes (= feature_map)?
                feature_map = model.forward(inputs)

            saver = tf.train.Saver(var_list=tf.global_variables(scope='YoloV3'))
            load_ops = load_weights(tf.global_variables(scope='YoloV3'), weight_path)
            sess.run(load_ops)
            saver.save(sess, save_path=save_path)


def parse_anchors(paths):
    """ parse anchors (somehow related to the desired layers?)
        Args:
            paths (dict): dictionary of paths from yml file
        Returns:
            anchors (np.array(float)): shape [N, 2] containing the anchors of the yolov3 model
    """

    model_file_path = paths['local_detection_model']
    anchor_path = os.path.join(model_file_path, 'yolov3', 'yolov3_anchors.txt')
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])

    return anchors


def load_weights(var_list, yolov3_weights_file):
    """ loads and converts pre-trained yolo weights to tensorflow.
        Args:
            var_list: list of network variables in tensorflow model.
            yolov3_weights_file: name of the binary file containing yolo weights.
        Returns:
            assign_ops: assignments to tensorflow model with yolo weights
    """

    with open(yolov3_weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops
