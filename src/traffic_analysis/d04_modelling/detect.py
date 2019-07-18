import os
import numpy as np
import cv2
from src.traffic_analysis.d00_utils.data_retrieval import retrieve_detect_model_configs_from_s3, get_project_directory


def detect_objects_in_image(image_capture, params, paths):
    """ unifying function that defines the detected objects in an image
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
        Returns:
            bboxes(list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    conf_thresh = params['confidence_threshold']
    nms_thresh = params['iou_threshold']

    retrieve_detect_model_configs_from_s3(params, paths)
    output_layers = pass_image_through_nn(image_capture, params)
    boxes_unfiltered, label_idxs_unfiltered, confs_unfiltered = describe_best_detections(image_capture,
                                                                                         output_layers, conf_thresh)
    boxes, label_idxs, confs = reduce_overlapping_detections(boxes_unfiltered, label_idxs_unfiltered,
                                                             confs_unfiltered, conf_thresh, nms_thresh)
    labels = label_detections(params, label_idxs)

    return boxes, labels, confs


def populate_model(params):
    """ locate files that correspond to the detection model of choice
        Args:
            params (dict): dictionary of parameters from yml file
        Returns:
            config_file_path (str): file path to the configuration file
            weights_file_path (str): file path to the weights file
    """

    model = params['yolo_model']
    project_dir = get_project_directory()
    config_file_path = os.path.join(project_dir, 'data', '00_detection', model, model + '.cfg')
    weights_file_path = os.path.join(project_dir, 'data', '00_detection', model, model + '.weights')
    return config_file_path, weights_file_path


def populate_labels(params):
    """ report full list of object labels corresponding to detection model of choice
        Args:
            params (dict): dictionary of parameters from yml file

        Returns:
            labels (list(str)): list of object labels strings
    """

    model = params['yolo_model']
    project_dir = get_project_directory()
    labels_file_path = os.path.join(project_dir, 'data', '00_detection', model, 'coco.names')
    f = open(labels_file_path, 'r')
    labels = [line.strip() for line in f.readlines()]

    return labels


def get_output_layers(net):
    """ (taken from cvlib) grabs another layer from output of object detection?
        Args:
            net (opencv.dnn): deep neural network created by opencv

        Returns:
            output_layers (list(str)): selected layers of neural network to be used in forward pass
    """

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def make_bbox_around_object(image_capture, unscaled_bbox):
    """ makes bounding boxes around detected objects at original scale of image
        Args:
            unscaled_bbox (nparray): nparray with unscaled width, height, and bottom-left coordinates of detected object
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)

        Returns:
            scaled_bbox (list(int)): width, height, and bottom-left coordinates of bounding box
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


def identify_most_probable_object(grid_cell):
    """ finds the most likely object to exist in a specific grid cell of image
        Args:
            grid_cell (nparray): nparray with scores of object labels in grid cell

        Returns:
            most_probable_object_idx (int): index of label of most probable object
            most_probable_object_score (float): score (i.e., confidence) of most probable object detected in image
    """

    scores = grid_cell[5:]  # ignore the physical parameters
    most_probable_object_idx = np.argmax(scores)
    most_probable_object_score = scores[most_probable_object_idx]

    return most_probable_object_idx, most_probable_object_score


def pass_image_through_nn(image_capture, params):
    """ detection model generates scores (i.e., confidence) of each object existing in image
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file

        Returns:
            output_layers (list(nparray)): list of neural network output layers and scores of predicted objects
    """

    # import classification model
    config, weights = populate_model(params)

    # convert image to "blob"
    scale = 0.00392  # required scaling for yolo
    blob = cv2.dnn.blobFromImage(image_capture, scale, (416, 416), (0, 0, 0), True, crop=False)

    # read model as deep neural network in opencv
    net = cv2.dnn.readNet(weights, config)  # can use other net, see documentation

    # input blob to neural network
    net.setInput(blob)

    # forward pass of blob through neural network
    output_layers = net.forward(get_output_layers(net))

    return output_layers


def describe_best_detections(image_capture, output_layers, conf_thresh):
    """ describes the detections that score above the confidence threshold
        Args:
            image_capture (nparray): numpy array containing the captured image (width, height, rbg)
            output_layers (list(nparray)): list of neural network outputs and scores of predicted objects
            conf_thresh (float): minimum confidence required in object detection, between 0 and 1
        Returns:
            bboxes (list(list(int))): list of width, height, and bottom-left coordinates of detection bounding boxes
            label_idxs (list(int)): list of indices corresponding to the detection labels
            confs (list(float)): list of scores of detections
    """

    # initialize the output lists
    bboxes = []
    label_idxs = []
    confs = []

    for output_layer in output_layers:  # loop through output layers from pass thru neural network
        for grid_cell in output_layer:  # loop through grid cells in output layer

            # find most likely object in specific grid cell of image
            object_label_idx, max_conf = identify_most_probable_object(grid_cell)

            # append object to running list of objects if prediction score is above confidence threshold
            if max_conf > conf_thresh:
                object_bbox = make_bbox_around_object(image_capture, grid_cell)
                bboxes.append(object_bbox)
                label_idxs.append(object_label_idx)
                confs.append(float(max_conf))

    return bboxes, label_idxs, confs


def reduce_overlapping_detections(bboxes_in, label_idxs_in, confs_in, conf_thresh, nms_thresh):
    """ removes the detections that score above the nms threshold
        Args:
            bboxes_in (list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            label_idxs_in (list(int)): list of indices corresponding to the detection labels
            confs_in (list(float)): list of scores of detections
            conf_thresh (float): minimum confidence required in object detection, between 0 and 1
            nms_thresh: non maximum suppression (nms) threshold to select for maximum overlap allowed between bboxes
        Returns:
            bboxes_out (list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            label_idxs_out (list(int)): list of indices corresponding to the detection labels
            confs_out (list(float)): list of detection scores
    """

    # report the indices of boxes that are screened through nms check
    idx_boxes_nms = cv2.dnn.NMSBoxes(bboxes_in, confs_in,
                                     conf_thresh, nms_thresh)

    # initialize output lists
    bboxes_out = []
    label_idxs_out = []
    confs_out = []

    # select "ins" of the reported indices
    for i in idx_boxes_nms:
        i = i[0]
        bbox = bboxes_in[i]
        w, h, x, y = bbox[:4]
        bboxes_out.append([round(x), round(y), round(x + w), round(y + h)])
        label_idxs_out.append(label_idxs_in[i])
        confs_out.append(confs_in[i])

    return bboxes_out, label_idxs_out, confs_out


def label_detections(params, label_idxs):
    """ labels the detected objects according to their index in list
        Args:
            params (dict): dictionary of parameters from yml file
            label_idxs (list(int)): list of indices corresponding to the detection labels
        Returns:
            labels (list(str)): labels of the reported object detections
    """

    # import the list of labels
    label_list = populate_labels(params)

    # initialize the output list
    labels = []

    # select label names according to indices selected
    for tick in label_idxs:
        labels.append(str(label_list[tick]))

    return labels
