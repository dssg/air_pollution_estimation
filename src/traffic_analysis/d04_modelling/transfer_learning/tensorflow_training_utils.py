# coding: utf-8
# part of this is take from Gluon's repo:
# https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/presets/yolo.py

from __future__ import division, print_function

import numpy as np
import cv2
import sys
import random
import tensorflow as tf
from collections import Counter
from tensorflow.core.framework import summary_pb2

PY_VERSION = sys.version_info[0]
iter_cnt = 0


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    params:
        boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
        labels: [N] shape, int64 dtype.
        class_num: int64 num.
        anchors: [9, 4] shape, float32 dtype.
    '''
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # convert boxes form:
    # shape: [N, 2]
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight.
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + class_num), np.float32)

    # mix up weight default to 1.
    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                         1] + 1e-10)
    # [N]
    best_match_idx = np.argmax(iou, axis=1)

    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        feature_map_group = 2 - idx // 3
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print(feature_map_group, '|', y,x,k,c)

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return y_true_13, y_true_26, y_true_52


def parse_data(line, class_num, img_size, anchors, mode, letterbox_resize):
    '''
    param:
        line: a line from the training/test txt file
        class_num: totol class nums.
        img_size: the size of image to be resized to. [width, height] format.
        anchors: anchors.
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
        letterbox_resize: whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
    '''
    if not isinstance(line, list):
        img_idx, pic_path, boxes, labels, _, _ = parse_line(line)
        img = cv2.imread(pic_path)
        # expand the 2nd dimension, mix up weight default to 1.
        boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    else:
        # the mix up case
        _, pic_path1, boxes1, labels1, _, _ = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, boxes2, labels2, _, _ = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, boxes = mix_up(img1, img2, boxes1, boxes2)
        labels = np.concatenate((labels1, labels2))

    if mode == 'train':
        # random color jittering
        # NOTE: applying color distort may lead to bad performance sometimes
        img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, boxes = random_expand(img, boxes, 4)

        # random cropping
        h, w, _ = img.shape
        boxes, crop = random_crop_with_constraints(boxes, (w, h))
        x0, y0, w, h = crop
        img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=interp, letterbox=letterbox_resize)

        # random horizontal flip
        h, w, _ = img.shape
        img, boxes = random_flip(img, boxes, px=0.5)
    else:
        img, boxes = resize_with_bbox(img, boxes, img_size[0], img_size[1], interp=1, letterbox=letterbox_resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


def get_batch_data(batch_line, class_num, img_size, anchors, mode, multi_scale=False, mix_up=False, letterbox_resize=True, interval=10):
    '''
    generate a batch of imgs and labels
    param:
        batch_line: a batch of lines from train/val.txt files
        class_num: num of total classes.
        img_size: the image size to be resized to. format: [width, height].
        anchors: anchors. shape: [9, 2].
        mode: 'train' or 'val'. if set to 'train', data augmentation will be applied.
        multi_scale: whether to use multi_scale training, img_size varies from [320, 320] to [640, 640] by default. Note that it will take effect only when mode is set to 'train'.
        letterbox_resize: whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
        interval: change the scale of image every interval batches. Note that it's indeterministic because of the multi threading.
    '''
    global iter_cnt
    # multi_scale training
    if multi_scale and mode == 'train':
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], [], []

    # mix up strategy
    if mix_up and mode == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            if np.random.uniform(0, 1) < 0.5:
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines

    for line in batch_line:
        img_idx, img, y_true_13, y_true_26, y_true_52 = parse_data(line, class_num, img_size, anchors, mode, letterbox_resize)

        img_idx_batch.append(img_idx)
        img_batch.append(img)
        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_13_batch), np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch


def mix_up(img1, img2, bbox1, bbox2):
    '''
    return:
        mix_img: HWC format mix up image
        mix_bbox: [N, 5] shape mix up bbox, i.e. `x_min, y_min, x_max, y_mix, mixup_weight`.
    '''
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    mix_img = np.zeros(shape=(height, width, 3), dtype='float32')

    # rand_num = np.random.random()
    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))
    mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * rand_num
    mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - rand_num)

    mix_img = mix_img.astype('uint8')

    # the last element of the 2nd dimention is the mix up weight
    bbox1 = np.concatenate((bbox1, np.full(shape=(bbox1.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox2 = np.concatenate((bbox2, np.full(shape=(bbox2.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox = np.concatenate((bbox1, bbox2), axis=0)

    return mix_img, mix_bbox


def bbox_crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """Crop an image randomly with bounding box constraints.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)


def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def resize_with_bbox(img, bbox, new_width, new_height, interp=0, letterbox=False):
    '''
    Resize the image and correct the bbox accordingly.
    '''

    if letterbox:
        image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height, interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh

        return image_padded, bbox
    else:
        ori_height, ori_width = img.shape[:2]

        img = cv2.resize(img, (new_width, new_height), interpolation=interp)

        # xmin, xmax
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / ori_width * new_width
        # ymin, ymax
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / ori_height * new_height

        return img, bbox


def random_flip(img, bbox, px=0, py=0):
    '''
    Randomly flip the image and correct the bbox.
    param:
    px:
        the probability of horizontal flip
    py:
        the probability of vertical flip
    '''
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox


def random_expand(img, bbox, max_ratio=4, fill=0, keep_ratio=True):
    '''
    Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    param:
    max_ratio :
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill :
        The value(s) for padded borders.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.
    '''
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=img.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = img

    # correct bbox
    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def shuffle_and_overwrite(file_name):
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)


def update_dict(ori_dict, new_dict):
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict


def list_add(ori_list, new_list):
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    param:
        var_list: list of network variables.
        weights_file: name of the binary file.
    """
    with open(weights_file, "rb") as fp:
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


def config_learning_rate(args, global_step):
    if args.lr_type == 'exponential':
        lr_tmp = tf.train.exponential_decay(args.learning_rate_init, global_step, args.lr_decay_freq,
                                            args.lr_decay_factor, staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, args.lr_lower_bound)
    elif args.lr_type == 'cosine_decay':
        train_steps = (args.total_epoches - float(args.use_warm_up) * args.warm_up_epoch) * args.train_batch_num
        return args.lr_lower_bound + 0.5 * (args.learning_rate_init - args.lr_lower_bound) * \
            (1 + tf.cos(global_step / train_steps * np.pi))
    elif args.lr_type == 'cosine_decay_restart':
        return tf.train.cosine_decay_restarts(args.learning_rate_init, global_step,
                                              args.lr_decay_freq, t_mul=2.0, m_mul=1.0,
                                              name='cosine_decay_learning_rate_restart')
    elif args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')
    elif args.lr_type == 'piecewise':
        return tf.train.piecewise_constant(global_step, boundaries=args.pw_boundaries, values=args.pw_values,
                                           name='piecewise_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')


def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')


def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_cpu(y_pred, y_true, num_classes, calc_now=True, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                      max_boxes=max_boxes, score_thresh=score_thresh, iou_thresh=iou_thresh)

        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, iou_thresh=0.5, calc_now=True):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    This function will perform gpu operation on the GPU.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 80] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels, each from 0 to 79
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = sess.run(gpu_nms_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, image_ids, y_pred):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    image_id = image_ids[0]

    # keep the first dimension 1
    pred_boxes = y_pred[0][0:1]
    pred_confs = y_pred[1][0:1]
    pred_probs = y_pred[2][0:1]

    boxes, scores, labels = sess.run(gpu_nms_op,
                                     feed_dict={pred_boxes_flag: pred_boxes,
                                                pred_scores_flag: pred_confs * pred_probs})

    pred_content = []
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max = boxes[i]
        score = scores[i]
        label = labels[i]
        pred_content.append([image_id, x_min, y_min, x_max, y_max, score, label])

    return pred_content


gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min, x_max, y_max = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        x_max * resize_ratio + dw,
                                        y_max * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([x_min * new_width / ori_width,
                                        y_min * new_height / ori_height,
                                        x_max * new_width / ori_width,
                                        y_max * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0, 0, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                     exact number of boxes
              scores: shape of [-1,]
              max_boxes: representing the maximum of boxes to be selected by non_max_suppression
              iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]
