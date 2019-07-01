# # Explore Tracking Methods in OpenCV
# Source: https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
from __future__ import print_function
import numpy as np
import sys, os
import cv2
from random import randint
import yaml
import cvlib
import imutils
import imageio
import time


def create_tracker_by_name(tracker_type:str):
	"""Create tracker based on tracker name"""
	tracker_types = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

	if tracker_type == tracker_types[0]:
		tracker = cv2.TrackerBoosting_create()
	elif tracker_type == tracker_types[1]:
		tracker = cv2.TrackerMIL_create()
	elif tracker_type == tracker_types[2]:
		tracker = cv2.TrackerKCF_create()
	elif tracker_type == tracker_types[3]:
		tracker = cv2.TrackerTLD_create()
	elif tracker_type == tracker_types[4]:
		tracker = cv2.TrackerMedianFlow_create()
	elif tracker_type == tracker_types[5]:
		tracker = cv2.TrackerGOTURN_create()
	elif tracker_type == tracker_types[6]:
		tracker = cv2.TrackerMOSSE_create()
	elif tracker_type == tracker_types[7]:
		tracker = cv2.TrackerCSRT_create()
	else:
		tracker = None
		print('Incorrect tracker name')
		print('Available trackers are:')
		for t in tracker_types:
			print(t)

	return tracker


def draw_bounding_boxes(frame:np.ndarray) -> (list,list):
	"""Select boxes by hand 


	"""
	bboxes = []
	colors = []

	# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
	# So we will call this function in a loop till we are done selecting all objects
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


def detect_bboxes(frame:np.ndarray, model:str,confidence:float,
						  implementation:str=None) -> (list,list,list):
	'''Detect bounding boxes on a frame using specified model and optionally an implementation

	bboxes returned in format (xmin, ymin, w, h)
	'''

	#todo: append confs to labels
	if implementation == 'cvlib':
		bboxes_cvlib, labels, confs = cvlib.detect_common_objects(frame, confidence=confidence,
											 model=model)
		bboxes = [bboxcvlib_to_bboxcv2(bbox_cvlib) for bbox_cvlib in bboxes_cvlib]

	del_inds = []
	label_confs = [] 
	for i, label in enumerate(labels): 
		#specify object types to ignore 
		if label not in ["car", "truck", "bus", "motorbike"]: del_inds.append(i)
		else: label_confs.append(label +' ' + str(format(confs[i] * 100, '.2f')) + '%') #append confidences to labels we care about

	#delete items from lists
	for i in sorted(del_inds, reverse=True):
		del bboxes[i]; del labels[i]; del confs[i]

	colors=color_bboxes(labels)
	return bboxes, label_confs, colors


def bboxcvlib_to_bboxcv2(bbox_cvlib):
	"""Convert bboxes from format returned by cvlib to format required by cv2

	"""
	xmin,ymin,xmin_plus_w, ymin_plus_h = bbox_cvlib[0], bbox_cvlib[1], bbox_cvlib[2], bbox_cvlib[3]
	bbox_cv2 = (xmin, ymin, xmin_plus_w - xmin, ymin_plus_h - ymin)
	return bbox_cv2


def bboxcv2_to_bboxcvlib(bbox_cv2):
	"""Convert bboxes from format (xmin,ymin,w,h) to format (xmin,ymin, xmin+w, ymin+H)
	"""
	xmin,ymin,w, h = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
	bbox_cvlib =(xmin, ymin, xmin+w, ymin+h)
	return bbox_cvlib


def color_bboxes(labels:list) -> list:
	"""auto generate colors based on object types

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

def bb_intersection_over_union(boxA, boxB) -> float:
	"""compute intersection over union for two bounding boxes
	boxes in format: (xmin, ymin, xmin+w, ymin+h)
	"""

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0]) #xcoords
	yA = max(boxA[1], boxB[1]) #ycoords
	xB = min(boxA[2], boxB[2]) #xcoords plus w
	yB = min(boxA[3], boxB[3]) #ycoords plus h

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
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


def determine_new_bboxes(bboxes_tracked:list, bboxes_detected:list, iou_threshold:float) -> list: 
	"""Return the indices of "new" bboxes in bboxes_detected so that a new tracker can be added for that 
	boxes should be passed in in format (xmin,ymin,w,h)
	"""
	new_bboxes_inds = set(range(len(bboxes_detected))) #init with all inds
	old_bboxes_inds = set()
	for i,boxA in enumerate(bboxes_detected): 
		#if find a box which has high IOU with an already-tracked box, consider it an old box
		for boxB in bboxes_tracked:
			#format conversion needed
			iou = bb_intersection_over_union(bboxcv2_to_bboxcvlib(boxA), bboxcv2_to_bboxcvlib(boxB))
			if iou > iou_threshold: #assume bbox has already been tracked and is not new 
				old_bboxes_inds.add(i) #add to set
	new_bboxes_inds=list(new_bboxes_inds.difference(old_bboxes_inds))
	return new_bboxes_inds


def track_objects(local_mp4_path:str,
				  detection_model:str,detection_confidence:float,detection_implementation:str,
				  detection_frequency:int,tracking_model:str,iou_threshold:float,
				  video_time_length=10, make_video=True):
	"""
	"""

	start_time = time.time()
	
	# Create a video capture object to read videos
	cap = cv2.VideoCapture(local_mp4_path)
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap_fps = int(n_frames / video_time_length)  # assumes vid_length in seconds

	# Read first frame
	success, frame = cap.read()

	if not success:
		print('Failed to read video')
		sys.exit(0)

	bboxes, label_confs, colors = detect_bboxes(frame=frame, model=detection_model,
										confidence=detection_confidence,
										implementation=detection_implementation)


	# Create MultiTracker object
	multiTracker = cv2.MultiTracker_create()

	# Initialize MultiTracker
	for bbox in bboxes:
		multiTracker.add(create_tracker_by_name(tracking_model), frame, bbox)

	processed_video = []

	# Process video and track objects
	frame_counter = 0
	while cap.isOpened():
		success, frame = cap.read()
		if not success: break

		# get updated location of objects in subsequent frames
		success, bboxes_tracked = multiTracker.update(frame)

		# draw tracked objects
		for i, tracked_box in enumerate(bboxes_tracked):
			p1 = (int(tracked_box[0]), int(tracked_box[1]))
			p2 = (int(tracked_box[0] + tracked_box[2]), int(tracked_box[1] + tracked_box[3]))
			cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
			#write labels, confs
			cv2.putText(frame, label_confs[i],(p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

		#every x frames, re-detect boxes
		if frame_counter%detection_frequency == 0: 
			#redetect bounding boxes
			bboxes_detected, label_confs_detected, colors_detected = detect_bboxes(frame=frame, model=detection_model,
							confidence=detection_confidence,
							implementation=detection_implementation)

			# re-initialize MultiTracker
			new_bbox_inds = determine_new_bboxes(bboxes_tracked, bboxes_detected, iou_threshold)
			# print(label_confs_detected)
			new_bboxes, new_label_confs,new_colors = [bboxes_detected[i] for i in new_bbox_inds],\
													 [label_confs_detected[i] for i in new_bbox_inds],\
													 [colors_detected[i] for i in new_bbox_inds]
			#iterate through new bboxes										 
			for i,new_bbox in enumerate(new_bboxes):
				multiTracker.add(create_tracker_by_name(tracking_model), frame, new_bbox)
			#add new colors and label confidences too 
			bboxes += new_bboxes; label_confs += new_label_confs; colors += new_colors

		processed_video.append(frame)

		# cv2.imshow('MultiTracker', frame)
		# # quit on ESC button
		# if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
		# 	break
		# frame_counter += 1

	if make_video:
		local_mp4_path_out = local_mp4_path[:-4] + '_trackedv4' + local_mp4_path[-4:]
		imageio.mimwrite(local_mp4_path_out, np.array(processed_video), fps=cap_fps)

	print('Run time is %s seconds' % (time.time() - start_time))
	return bboxes, label_confs #ideally would return number of tracked objects, confidence levels


if __name__ == '__main__':
	#config stuff
	basepath=os.path.dirname(__file__) #path of current script

	os.chdir(".")
	with open('../../conf/base/parameters.yml') as f:
	   params = yaml.safe_load(f)['modelling']
	os.chdir(".")
	with open('../../conf/base/paths.yml') as f:
	   paths = yaml.safe_load(f)['paths']

	#get a video from local
	local_mp4_path = os.path.abspath(os.path.join(basepath,"..", "..","data/sample_video_data/testvid.mp4"))
	tracking_model = params["opencv_tracker_type"]
	detection_confidence = params["detection_confidence_threshold"]
	detection_model = params["yolo_model"]
	detection_implementation = params["yolo_implementation"]
	detection_frequency = 75
	iou_threshold = params["iou_threshold"]

	bboxes, label_confs = track_objects(local_mp4_path=local_mp4_path,
				  detection_model=detection_model,detection_confidence=detection_confidence,
				  detection_implementation=detection_implementation,
				  detection_frequency=detection_frequency,tracking_model=tracking_model,
				  iou_threshold=iou_threshold, make_video=True)

	print(label_confs)