# coding: utf-8

from __future__ import division, print_function
import os
import math
import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

from traffic_analysis.d04_modelling.transfer_learning.tensorflow_training_utils import get_batch_data, \
    make_summary, config_learning_rate, config_optimizer, AverageMeter, \
    evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec, gpu_nms
from traffic_analysis.d04_modelling.transfer_learning.tensorflow_model_loader import YoloV3
from traffic_analysis.d04_modelling.transfer_learning.convert_darknet_to_tensorflow import parse_anchors
from traffic_analysis.d04_modelling.transfer_learning.tensorflow_detection_utils import read_class_names


def transfer_learn(paths, params, train_params, train_file, test_file, selected_labels):
    """ trains last three layers of yolov3 network on custom dataset
    """

    transfer_learn_model_dir = os.path.join(paths['local_detection_model'], train_params['trained_model_name'])
    if not os.path.exists(transfer_learn_model_dir):
        os.makedirs(transfer_learn_model_dir)

    truth_dir_path = paths['temp_annotation']
    class_name_path = os.path.join(paths['local_detection_model'], 'yolov3', 'coco.names')  # CHANGE THIS
    classes = read_class_names(class_name_path)

    selected_label_idxs = []
    for idx, label in classes.items():
        if label in selected_labels:
            selected_label_idxs.append(idx)
    anchors = parse_anchors(paths)
    number_classes = len(classes)
    
    train_data_path = os.path.join(truth_dir_path, train_file)
    test_data_path = os.path.join(truth_dir_path, test_file)
    train_img_cnt = len(open(train_data_path, 'r').readlines())
    val_img_cnt = len(open(test_data_path, 'r').readlines())
    train_batch_num = int(math.ceil(float(train_img_cnt) / train_params['num_batches']))

    lr_decay_freq = int(train_batch_num * train_params['lr_decay_epoch'])

    logging_file_path = os.path.join(truth_dir_path, 'progress.log')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filename=logging_file_path, filemode='w')

    is_training = tf.placeholder(tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
    gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, number_classes, train_params['nms_topk'],
                         train_params['score_threshold'], train_params['nms_threshold'])

    train_dataset = tf.data.TextLineDataset(train_data_path)
    train_dataset = train_dataset.shuffle(train_img_cnt)
    train_dataset = train_dataset.batch(train_params['num_batches'])
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, number_classes, [416, 416], anchors, 'train', True, True, True],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=train_params['num_threads'])
    train_dataset = train_dataset.prefetch(train_params['prefetech_buffer'])
    
    test_dataset = tf.data.TextLineDataset(test_data_path)
    test_dataset = test_dataset.batch(1)
    test_dataset = test_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, number_classes, [416, 416], anchors, 'val', False, False, True],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=train_params['num_threads'])
    test_dataset.prefetch(train_params['prefetech_buffer'])
    
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(test_dataset)
    
    # get an element from the chosen dataset iterator
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]
    
    # tf.data pipeline will lose the data `static` shape, so we need to set it manually
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])
    
    # define model
    yolo_model = YoloV3(number_classes, anchors, use_label_smooth=True, use_focal_loss=True,
                        batch_norm_decay=train_params['batch_norm_decay'], weight_decay=train_params['weight_decay'],
                        use_static_shape=False)
    
    with tf.variable_scope('YoloV3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
    loss = yolo_model.compute_loss(pred_feature_maps, y_true)
    y_pred = yolo_model.predict(pred_feature_maps)
    
    l2_loss = tf.losses.get_regularization_loss()
    
    # setting restore parts and vars to update
    saver_to_restore = tf.train.Saver(
        var_list=tf.contrib.framework.get_variables_to_restore(
            include=None,
            exclude=['YoloV3/yolov3_head/Conv_14', 'YoloV3/yolov3_head/Conv_6', 'YoloV3/yolov3_head/Conv_22']))
    update_vars = tf.contrib.framework.get_variables_to_restore(include=['YoloV3/yolov3_head'])
    
    tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
    tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
    tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
    tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
    tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
    tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])
    
    global_step = tf.Variable(float(train_params['global_step']),
                              trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    
    learning_rate = tf.cond(tf.less(global_step, train_batch_num * train_params['warm_up_epoch']),
                            lambda: train_params['learning_rate_init'] *
                                    global_step / (train_batch_num * train_params['warm_up_epoch']),
                            lambda: config_learning_rate(lr_decay_freq=lr_decay_freq, train_batch_num=train_batch_num,
                                                         global_step=global_step -
                                                         train_batch_num * train_params['warm_up_epoch']))
    tf.summary.scalar('learning_rate', learning_rate)
    
    if not train_params['save_optimizer']:
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()
    
    optimizer = config_optimizer(train_params['optimizer_name'], learning_rate)
    
    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss[0] + l2_loss, var_list=update_vars, global_step=global_step)
        # apply gradient clip to avoid gradient exploding
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        clip_grad_var = [gv if gv[0] is None else [
              tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)
    
    if train_params['save_optimizer']:
        print('Saving optimizer parameters to checkpoint! Remember to restore global_step in fine-tuning afterwards.')
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()
    
    tensorboard_log_path = os.path.join(truth_dir_path, 'tensorboard_logs')
    yolov3_tensorflow_path = os.path.join(paths['local_detection_model'], params['detection_model'], 'yolov3.ckpt')
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver_to_restore.restore(sess, yolov3_tensorflow_path)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_log_path, sess.graph)
    
        print('\n----------- start to train -----------\n')
    
        best_mAP = -np.Inf
    
        for epoch in range(train_params['total_epochs']):
    
            sess.run(train_init_op)
            loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), \
                                                                  AverageMeter(), AverageMeter()
    
            for i in trange(train_batch_num):
                _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                    [train_op, merged, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True})
    
                writer.add_summary(summary, global_step=__global_step)
    
                loss_total.update(__loss[0], len(__y_pred[0]))
                loss_xy.update(__loss[1], len(__y_pred[0]))
                loss_wh.update(__loss[2], len(__y_pred[0]))
                loss_conf.update(__loss[3], len(__y_pred[0]))
                loss_class.update(__loss[4], len(__y_pred[0]))
    
                if __global_step % train_params['train_evaluation_step'] == 0 and __global_step > 0:
                    recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag,
                                                        __y_pred, __y_true, number_classes,
                                                        train_params['nms_threshold'])
    
                    info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, " \
                           "wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(epoch, int(__global_step),
                                                                               loss_total.average, loss_xy.average,
                                                                               loss_wh.average, loss_conf.average,
                                                                               loss_class.average)
                    info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall, precision, __lr)
                    print(info)
                    logging.info(info)
    
                    writer.add_summary(make_summary('evaluation/train_batch_recall', recall),
                                       global_step=__global_step)
                    writer.add_summary(make_summary('evaluation/train_batch_precision', precision),
                                       global_step=__global_step)
    
                    if np.isnan(loss_total.average):
                        print('****' * 10)
                        raise ArithmeticError(
                            'Gradient exploded! Please train again and you may need modify some parameters.')
    
            # NOTE: this is just demo. You can set the conditions when to save the weights.
            if epoch % train_params['save_epoch'] == 0 and epoch > 0:
                if loss_total.average <= 2.:
                    saver_to_save.save(sess,
                                       os.path.join(train_params['trained_model_name'],
                                                    'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(
                                                        epoch, int(__global_step), loss_total.average, __lr)))
    
            # switch to validation dataset for evaluation
            if epoch % train_params['val_evaluation_epoch'] == 0 and epoch >= train_params['warm_up_epoch']:
                sess.run(val_init_op)
    
                val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                    AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
                val_preds = []
    
                for j in trange(val_img_cnt):
                    __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss],
                                                             feed_dict={is_training: False})
                    pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag,
                                                 pred_scores_flag, __image_ids, __y_pred)
                    val_preds.extend(pred_content)
                    val_loss_total.update(__loss[0])
                    val_loss_xy.update(__loss[1])
                    val_loss_wh.update(__loss[2])
                    val_loss_conf.update(__loss[3])
                    val_loss_class.update(__loss[4])
    
                # calc mAP
                rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                gt_dict = parse_gt_rec(test_data_path, [416, 416], letterbox_resize=True)
    
                info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

                for class_idx in range(number_classes):
                    if class_idx in selected_label_idxs:
                        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, class_idx,
                                                           iou_thres=train_params['eval_threshold'],
                                                           use_07_metric=True)
                        info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(class_idx,
                                                                                                         rec, prec, ap)

                        if math.isnan(rec) or math.isnan(prec) or math.isnan(ap):
                            pass
                        else:
                            rec_total.update(rec, npos)
                            prec_total.update(prec, nd)
                            ap_total.update(ap, 1)

                mAP = ap_total.average
                info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(
                    rec_total.average, prec_total.average, mAP)
                info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                    val_loss_total.average, val_loss_xy.average, val_loss_wh.average,
                    val_loss_conf.average, val_loss_class.average)
                print(info)
                logging.info(info)
    
                if mAP > best_mAP:
                    best_mAP = mAP
                    saver_best.save(sess, os.path.join(
                        transfer_learn_model_dir,
                        'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                            epoch, int(__global_step), best_mAP, val_loss_total.average, __lr)))
    
                writer.add_summary(make_summary('evaluation/val_mAP', mAP),
                                   global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_recall', rec_total.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', prec_total.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average),
                                   global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average),
                                   global_step=epoch)
                
    return
