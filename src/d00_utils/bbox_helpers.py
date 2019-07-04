import numpy as np 
import cv2
from random import randint


def manually_draw_bboxes(frame:np.ndarray) -> (list,list):
	"""Select boxes by hand. If this is called in a while loop, do NOT press c to cancel 
	selection (this somehow messes up the selection process). Assigns random colors to bboxes
	"""
	bboxes,colors = [],[]

	# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
	# So you can call this function in a loop till you are done selecting all objects
	while True:
		# draw bounding boxes over objects
		# selectROI's default behaviour is to draw box starting from the center
		# when fromCenter is set to false, you can draw box starting from top left corner
		bbox = cv2.selectROI('MultiTracker', frame)
		bboxes.append(bbox)
		colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
		print("Press q to quit selecting boxes and start tracking. Press any other key to select next object.")
		k = cv2.waitKey(0) & 0xFF
		if (k == 113):  # q is pressed
			break

		print('Selected bounding boxes {}'.format(bboxes))

	return bboxes,colors


def bboxcvlib_to_bboxcv2(bbox_cvlib):
	"""Convert bboxes from format returned by cvlib (xmin,ymin, xmin+w, ymin+H)
	to format required by cv2 (xmin,ymin,w,h)
	"""
	xmin,ymin,xmin_plus_w, ymin_plus_h = bbox_cvlib[0], bbox_cvlib[1], bbox_cvlib[2], bbox_cvlib[3]
	bbox_cv2 = (xmin, ymin, xmin_plus_w - xmin, ymin_plus_h - ymin)
	return bbox_cv2


def bboxcv2_to_bboxcvlib(bbox_cv2):
	"""Convert bboxes from format returned by cv2 (xmin,ymin,w,h) 
	to format accepted by cvlib (xmin,ymin, xmin+w, ymin+H)
	"""
	xmin,ymin,w, h = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
	bbox_cvlib =(xmin, ymin, xmin+w, ymin+h)
	return bbox_cvlib


def color_bboxes(labels:list) -> list:
	"""Based on object types in the list, will return a color for that object. 
	If color is not in the dict, random color will be generated. 

	Keyword arguments 

	labels: list of strings (types of objects)
	"""
	color_dict={"car": (255,100,150), #pink
				"truck": (150,230,150), #light green
				 "bus": (150,200,230), #periwinkle
				 "motorbike": (240,160,80)} #orange
	colors = []
	for label in labels:
		if label not in color_dict.keys():
			color_dict[label]= (randint(0,255), randint(0,255), randint(0,255))
		colors.append(color_dict[label])
	return colors


def bbox_intersection_over_union(boxA, boxB) -> float:
	"""Compute intersection over union for two bounding boxes

    Keyword arguments: 

	boxA -- format is (xmin, ymin, xmin+w, ymin+h) 
	boxB -- format is (xmin, ymin, xmin+w, ymin+h)
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0]) #xcoords
	yA = max(boxA[1], boxB[1]) #ycoords
	xB = min(boxA[2], boxB[2]) #xcoords plus w
	yB = min(boxA[3], boxB[3]) #ycoords plus h

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA), 0) * max((yB - yA), 0))
	if interArea == 0:
		return 0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
	boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def vectorized_intersection_over_union(bboxes_t0:np.ndarray, bboxes_t1:np.ndarray) ->np.ndarray:
	""" This function uses np vectorized operations to compute the iou for sets of vehicles 
	2d arrays 
	boxA -- format is (xmin, ymin, xmin+w, ymin+h)
	"""
	assert bboxes_t0.shape[1] == 4 and bboxes_t1.shape[1] == 4 

	xA = np.maximum(bboxes_t0[:,0], bboxes_t1[:,0])
	yA = np.maximum(bboxes_t0[:,1], bboxes_t1[:,1])
	xB = np.maximum(bboxes_t0[:,2], bboxes_t1[:,2])
	yB = np.maximum(bboxes_t0[:,3], bboxes_t1[:,3])

	interArea = np.abs(np.multiply(np.maximum(xB - xA,0), np.maximum(yB - yA, 0)))

	boxAArea = np.abs(np.multiply((bboxes_t0[:,2] - bboxes_t0[:,0]) , (bboxes_t0[:,3] - bboxes_t0[:,1])))
	boxBArea = np.abs(np.multiply((bboxes_t1[:,2] - bboxes_t1[:,0]), (bboxes_t1[:,3] - bboxes_t1[:,1])))

	unionArea = boxAArea + boxBArea - interArea
	
	with np.errstate(divide='ignore'):
		iou = interArea / unionArea

	return iou


def display_bboxes_on_frame(frame:np.ndarray, bboxes:list, colors:list, box_labels:list):
	"""Draw bounding boxes on a frame using provided colors, and displays labels/confidences 

	Keyword arguments 

	bboxes: provide in cv2 format (xmin,ymin, w, h)
	colors: list of RGB tuples 
	box_labels: list of strings with which to label each box
	"""
	for i, box in enumerate(bboxes):
		p1 = (int(box[0]), int(box[1]))
		p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
		cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
		#write labels, confs
		cv2.putText(frame, box_labels[i],(p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
	return 

