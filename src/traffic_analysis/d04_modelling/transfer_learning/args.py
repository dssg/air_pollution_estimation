# coding: utf-8
# This file contains the parameter used in train.py

## EVENTUALLY CHANGE THIS TO YAML

from __future__ import division, print_function

from traffic_analysis.d04_modelling.transfer_learning.tensorflow_training_utils import parse_anchors, read_class_names
from traffic_analysis.d00_utils.get_project_directory import get_project_directory
import math
import os

# paths
project_dir = get_project_directory
local_model_file_path = os.path.join(project_dir, 'data', 'ref', 'detection_model')
local_annotation_file_path = os.path.join(project_dir, 'data', 'ref', 'annotations')

train_file = os.path.join(local_annotation_file_path, 'transfer_learning', 'TRAIN.TXT')
val_file = os.path.join(local_annotation_file_path, 'transfer_learning', 'VAL.TXT')
restore_path = os.path.join(local_model_file_path, 'yolo_tf', 'yolov3.ckpt')
save_dir = os.path.join(local_model_file_path, 'vehicle_detector')  # directory where weights are saved
log_dir = os.path.join(local_model_file_path, 'vehicle_detector', 'logs')  # The directory to store the log files.
progress_log_path = os.path.join(local_model_file_path, 'vehicle_detector', 'logs', 'progress.log')  # training progress
anchor_path = os.path.join(local_model_file_path, 'yolov3', 'yolo_anchors.txt')  # The path of the anchor txt file.
class_name_path = os.path.join(local_model_file_path, 'vehicle_detector', 'COCO.NAMES')  # The path of the class names.

### Training releated numbers
batch_size = 6
img_size = [416, 416]  # Images will be resized to `img_size` and fed to the network, size format: [width, height]
letterbox_resize = True  # Whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
total_epoches = 100
train_evaluation_step = 100  # Evaluate on the training batch after some steps.
val_evaluation_epoch = 2  # Evaluate on the whole validation dataset after some epochs. Set to None to evaluate every epoch.
save_epoch = 10  # Save the model after some epochs.
batch_norm_decay = 0.99  # decay in bn ops
weight_decay = 5e-4  # l2 weight decay
global_step = 0  # used when resuming training

### tf.data parameters
num_threads = 10  # Number of threads for image processing used in tf.data pipeline.
prefetech_buffer = 5  # Prefetech_buffer used in tf.data pipeline.

### Learning rate and optimizer
optimizer_name = 'momentum'  # Chosen from [sgd, momentum, adam, rmsprop]
save_optimizer = True  # Whether to save the optimizer parameters into the checkpoint file.
learning_rate_init = 1e-4
lr_type = 'piecewise'  # Chosen from [fixed, exponential, cosine_decay, cosine_decay_restart, piecewise]
lr_decay_epoch = 5  # Epochs after which learning rate decays. Int or float. Used when chosen `exponential` and `cosine_decay_restart` lr_type.
lr_decay_factor = 0.96  # The learning rate decay factor. Used when chosen `exponential` lr_type.
lr_lower_bound = 1e-6  # The minimum learning rate.
# only used in piecewise lr type
pw_boundaries = [30, 50]  # epoch based boundaries
pw_values = [learning_rate_init, 3e-5, 1e-5]

### Load and finetune
# Choose the parts you want to restore the weights. List form.
# restore_include: None, restore_exclude: None  => restore the whole model
# restore_include: None, restore_exclude: scope  => restore the whole model except `scope`
# restore_include: scope1, restore_exclude: scope2  => if scope1 contains scope2, restore scope1 and not restore scope2 (scope1 - scope2)
# choise 1: only restore the darknet body
# restore_include = ['yolov3/darknet53_body']
# restore_exclude = None
# choise 2: restore all layers except the last 3 conv2d layers in 3 scale
restore_include = None
restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
# Choose the parts you want to finetune. List form.
# Set to None to train the whole model.
update_part = ['yolov3/yolov3_head']

### other training strategies
multi_scale_train = True  # Whether to apply multi-scale training strategy. Image size varies from [320, 320] to [640, 640] by default.
use_label_smooth = True # Whether to use class label smoothing strategy.
use_focal_loss = True  # Whether to apply focal loss on the conf loss.
use_mix_up = True  # Whether to use mix up data augmentation strategy.
use_warm_up = True  # whether to use warm up strategy to prevent from gradient exploding.
warm_up_epoch = 3  # Warm up training epoches. Set to a larger value if gradient explodes.

### some constants in validation
# nms
nms_threshold = 0.45  # iou threshold in nms operation
# threshold of the probability of the classes in nms operation,
# i.e. score = pred_confs * pred_probs. set lower for higher recall.
score_threshold = 0.01
nms_topk = 150  # keep at most nms_topk outputs after nms
# mAP eval
eval_threshold = 0.5  # the iou threshold applied in mAP evaluation
use_voc_07_metric = False  # whether to use voc 2007 evaluation metric, i.e. the 11-point metric

### parse some params
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
class_num = len(classes)
train_img_cnt = len(open(train_file, 'r').readlines())
val_img_cnt = len(open(val_file, 'r').readlines())
train_batch_num = int(math.ceil(float(train_img_cnt) / batch_size))

lr_decay_freq = int(train_batch_num * lr_decay_epoch)
pw_boundaries = [float(i) * train_batch_num + global_step for i in pw_boundaries]