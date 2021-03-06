import numpy as np
import cv2
from random import randint


def bboxcvlib_to_bboxcv2(bbox_cvlib, vectorized=False):
    """Convert bboxes from format returned by cvlib (xmin,ymin, xmin+w, ymin+H)
    to format required by cv2 (xmin,ymin,w,h)

    Args:
        vectorized: if set to True, will handle np arrays of bboxes. 
                    Format would be (num_bboxes, 4)
    """
    if vectorized == False:  # handles subscriptable objects
        xmin, ymin, xmin_plus_w, ymin_plus_h = bbox_cvlib[
            0], bbox_cvlib[1], bbox_cvlib[2], bbox_cvlib[3]
        bbox_cv2 = [xmin, ymin, xmin_plus_w - xmin, ymin_plus_h - ymin]

    else:  # handles np arrays
        xmin, ymin, xmin_plus_w, ymin_plus_h = bbox_cvlib[:, 0], bbox_cvlib[:, 1],\
            bbox_cvlib[:, 2], bbox_cvlib[:, 3]
        bbox_cv2 = np.array([xmin, ymin,
                             xmin_plus_w - xmin,
                             ymin_plus_h - ymin]).transpose()

    return bbox_cv2


def bboxcv2_to_bboxcvlib(bbox_cv2,  vectorized=False):
    """Convert bboxes from format returned by cv2 (xmin,ymin,w,h)
    to format accepted by cvlib (xmin,ymin, xmin+w, ymin+H)

    Args: 
        vectorized: if set to True, will handle np arrays of bboxes. 
                    Format would be (num_bboxes, 4)
    """
    if vectorized == False:  # handles subscriptable items
        xmin, ymin, w, h = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
        bbox_cvlib = [xmin, ymin, xmin+w, ymin+h]

    else:  # handles np arrays with multiple bboxes
        xmin, ymin, w, h = bbox_cv2[:, 0], bbox_cv2[:,
                                                    1], bbox_cv2[:, 2], bbox_cv2[:, 3]
        bbox_cvlib = np.array([xmin, ymin, xmin+w, ymin+h]).transpose()

    return bbox_cvlib


def color_bboxes(labels: list) -> list:
    """Based on object types in the list, will return a color for that object.
    If color is not in the dict, random color will be generated.

    Args:
        labels: list of strings (types of objects)
    """
    color_dict = {"car": (255, 100, 150),  # pink
                  "truck": (150, 230, 150),  # light green
                  "bus": (150, 200, 230),  # periwinkle
                  "motorbike": (240, 160, 80)}  # orange
    colors = []
    for label in labels:
        if label not in color_dict.keys():
            color_dict[label] = (
                randint(0, 255), randint(0, 255), randint(0, 255))
        colors.append(color_dict[label])
    return colors


def bbox_intersection_over_union(bbox_a, bbox_b) -> float:
    """Compute intersection over union for two bounding boxes

    Args: 
        bbox_a -- format is (xmin, ymin, xmin+width, ymin+height)
        bbox_b -- format is (xmin, ymin, xmin+width, ymin+height)
    """
    assert (bbox_a[0] <= bbox_a[2] and bbox_a[1] <= bbox_a[3]
            ), f"Please make sure arg bbox_a is in format (xmin,ymin,xmin+w,ymin+h): {bbox_a}"
    assert (bbox_b[0] <= bbox_b[2] and bbox_b[1] <= bbox_b[3]
            ), f"Please make sure arg bbox_b is in in format (xmin,ymin,xmin+w,ymin+h): {bbox_b}"

    # determine the (x, y)-coordinates of the intersection rectangle
    x_upper_left = max(bbox_a[0], bbox_b[0])  # xcoords
    y_upper_left = max(bbox_a[1], bbox_b[1])  # ycoords
    x_lower_right = min(bbox_a[2], bbox_b[2])  # xcoords plus w
    y_lower_right = min(bbox_a[3], bbox_b[3])  # ycoords plus h

    # compute the area of intersection rectangle
    inter_area = abs(max((x_lower_right - x_upper_left, 0))
                     * max((y_lower_right - y_upper_left), 0))
    if inter_area == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    bbox_a_area = abs((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    bbox_b_area = abs((bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(bbox_a_area + bbox_b_area - inter_area)

    # return the intersection over union value
    return iou


def display_bboxes_on_frame(frame: np.ndarray, 
                            bboxes: list, 
                            colors: list, 
                            box_labels: list, 
                            display_text=True):
    """Draw bounding boxes on a frame using provided colors, and displays labels/confidences

    Args:
        bboxes: provide in cv2 format (xmin,ymin, width, height)
        colors: list of RGB tuples
        box_labels: list of strings with which to label each box
        display_text: display text
    """
    for i, box in enumerate(bboxes):
        pt_upper_left = (int(box[0]), int(box[1]))
        pt_lower_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(img=frame,
                      pt1=pt_upper_left,
                      pt2=pt_lower_right,
                      color=colors[i],
                      thickness=2,
                      lineType=1)
        # write labels, confs
        if display_text:
            cv2.putText(img=frame,
                        text=box_labels[i],
                        org=(pt_upper_left[0], pt_upper_left[1]-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=colors[i],
                        thickness=2)
