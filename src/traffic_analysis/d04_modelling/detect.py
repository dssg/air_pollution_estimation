import os
import numpy as np
import cv2
from src.traffic_analysis.d00_utils.load_confs import load_parameters, load_paths
from src.traffic_analysis.d00_utils.data_retrieval import retrieve_detect_model_configs_from_s3


def populate_model(params):
    """ locate files that correspond to the detection model of choice
        Args:
            params (dict): dictionary of parameters from yml file
        Returns:
            config_file_path (str): file path to the configuration file
            weights_file_path (str): file path to the weights file
    """

    model = params['yolo_model']
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
    config_file_path = os.path.join(project_dir, 'data', '00_detection', model, model + '.cfg')  # change if model isn't yolo
    weights_file_path = os.path.join(project_dir, 'data', '00_detection', model, model + '.weights')  # change if model isn't yolo
    return config_file_path, weights_file_path


def populate_labels(params):
    """ report full list of object labels corresponding to detection model of choice
        Args:
            params (dict): dictionary of parameters from yml file

        Returns:
            labels (list(str)): list of object labels strings
    """

    model = params['yolo_model']
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
    labels_file_path = os.path.join(project_dir, 'data', '00_detection', model, 'coco.names')  # change if not coco
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


def make_bbox_around_object(imcap, detection):
    """ makes bounding boxes around detected objects at original scale of image
        Args:
            detection (nparray): nparray with unscaled width, height, and bottom-left coordinates of detected object
            imcap (nparray): numpy array containing the captured image (width, height, rbg)

        Returns:
            bbox (list(int)): width, height, and bottom-left coordinates of bounding box
    """

    imheight, imwidth = imcap.shape[:2]
    center_x = int(detection[0] * imwidth)
    center_y = int(detection[1] * imheight)
    w = int(detection[2] * imwidth)
    h = int(detection[3] * imheight)
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    bbox = [w, h, x, y]

    return bbox


def identify_most_probable_object(detection):
    """ finds the most likely object to exist in a specific grid cell of image
        Args:
            detection (nparray): nparray with scores of object labels in grid cell

        Returns:
            most_probable_object_idx (int): index of label of most probable object
            max_score (float): score (i.e., confidence) of most probable object detected in image
    """

    scores = detection[5:]  # ignore the physical parameters
    most_probable_object_idx = np.argmax(scores)
    max_score = np.max(scores)

    return most_probable_object_idx, max_score


def predict_objects_in_image(imcap, params):
    """ detection model generates scores (i.e., confidence) of each object existing in image
        Args:
            imcap (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file

        Returns:
            predictions (list(nparray)): list of neural network outputs and scores of predicted objects
    """

    # import classification model
    config, weights = populate_model(params)

    # convert image to "blob"
    scale = 0.00392  # required scaling for yolo
    blob = cv2.dnn.blobFromImage(imcap, scale, (416, 416), (0, 0, 0), True, crop=False)

    # read model as deep neural network in opencv
    net = cv2.dnn.readNet(weights, config)  # can use other net, see documentation

    # input blob to neural network
    net.setInput(blob)

    # forward pass of blob through neural network
    predictions = net.forward(get_output_layers(net))

    return predictions


def describe_best_detections(imcap, predictions, conf_thresh):
    """ describes the detections that score above the confidence threshold
        Args:
            imcap (nparray): numpy array containing the captured image (width, height, rbg)
            predictions (list(nparray)): list of neural network outputs and scores of predicted objects
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

    for prediction in predictions:  # loop through prediction layers
        for detections in prediction:  # loop through class detection probabilities

            # find most likely object in specific grid cell of image
            object_label_idx, max_conf = identify_most_probable_object(detections)

            # append object to running list of objects if prediction score is above confidence threshold
            if max_conf > conf_thresh:
                object_bbox = make_bbox_around_object(imcap, detections)
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
    idx_boxes_nms = cv2.dnn.NMSBoxes(bboxes=bboxes_in, scores=confs_in,
                                     score_threshold=conf_thresh, nms_threshold=nms_thresh)

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


def detect_objects_in_image(imcap, params, paths):
    """ unifying function that defines the detected objects in an image
        Args:
            imcap (nparray): numpy array containing the captured image (width, height, rbg)
            params (dict): dictionary of parameters from yml file
        Returns:
            bboxes(list(list(int))): list of width, height, and bottom-left coordinates of detection bboxes
            labels (list(str)): list of detection labels
            confs (list(float)): list of detection scores
    """

    # define thresholds based on params file
    conf_thresh = params['confidence_threshold']
    nms_thresh = params['iou_threshold']

    # download model configuration files if not already on local from s3
    retrieve_detect_model_configs_from_s3(params, paths)

    # predict detected objects in the image
    predictions = predict_objects_in_image(imcap, params)

    # define detected objects based on confidences of detections
    boxes_unfiltered, label_idxs_unfiltered, confs_unfiltered = describe_best_detections(imcap,
                                                                                         predictions, conf_thresh)

    # reduce overlap between detections of same object with nms
    boxes, label_idxs, confs = reduce_overlapping_detections(boxes_unfiltered, label_idxs_unfiltered,
                                                             confs_unfiltered, conf_thresh, nms_thresh)

    # generate labels of final detections
    labels = label_detections(params, label_idxs)

    return boxes, labels, confs


if __name__ == '__main__':
    params = load_parameters()
    paths = load_paths()
    imcap = cv2.imread('C:/Users/joh3146/Documents/dssg/air_pollution_estimation/data/01_raw/jamcams/frame001.jpg')
    b, l, c = detect_objects_in_image(imcap, params, paths)
    print('yay')
