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
		bboxes = [bboxcvlib_to_bboxcv2(tuple(bbox_cvlib)) for bbox_cvlib in bboxes_cvlib]

	del_inds = []
	for i, label in enumerate(labels): 
		#specify object types to ignore 
		if label not in ["car", "truck", "bus", "motorbike"]: del_inds.append(i)
		else: labels[i] += ' ' + str(format(confs[i] * 100, '.2f')) + '%' #append confidences to labels we care about

	#delete items from lists
	for i in sorted(del_inds, reverse=True):
		del bboxes[i]; del labels[i]; del confs[i]

	return bboxes, labels, confs


def bboxcvlib_to_bboxcv2(bbox_cvlib:tuple) -> tuple:
	"""Convert bboxes from format returned by cvlib to format required by cv2

	"""
	x,y,x_plus_w, y_plus_h = bbox_cvlib[0], bbox_cvlib[1], bbox_cvlib[2], bbox_cvlib[3]
	bbox_cv2=(x, y, x_plus_w - x, y_plus_h-y)
	return bbox_cv2


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


def detect_new_bboxes(frame:np.ndarray,bboxes:list) -> list : 
	"""Determine new bboxes from old 

	"""

	pass
	return None


def track_objects(local_mp4_path:str,
				  detection_model:str,detection_confidence:float,detection_implementation:str,
				  detection_frequency:int,tracking_model:str,
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

	bboxes, labels, confs = detect_bboxes(frame=frame, model=detection_model,
							confidence=detection_confidence,
							implementation=detection_implementation)

	colors=color_bboxes(labels)

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
		success, updated_boxes = multiTracker.update(frame)

		# draw tracked objects
		for i, updated_box in enumerate(updated_boxes):
			p1 = (int(updated_box[0]), int(updated_box[1]))
			p2 = (int(updated_box[0] + updated_box[2]), int(updated_box[1] + updated_box[3]))
			cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
			#write labels, confs
			cv2.putText(frame, labels[i],(p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

		#every x frames, re-detect boxes
		if frame_counter%detection_frequency == 0: 
			#redetect bounding boxes
			bboxes, labels, confs = detect_bboxes(frame=frame, model=detection_model,
							confidence=detection_confidence,
							implementation=detection_implementation)
			#somehow only get bboxes with intersection over union under a certain amt? 
			colors=color_bboxes(labels)
			# Create MultiTracker object
			multiTracker = cv2.MultiTracker_create()

			# re-initialize MultiTracker
			new_bboxes = detect_new_bboxes(frame,bboxes)
			for bbox in bboxes:
				multiTracker.add(create_tracker_by_name(tracking_model), frame, bbox)


		processed_video.append(frame)

		# cv2.imshow('MultiTracker', frame)
		# quit on ESC button
		# if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
		# 	break
		frame_counter += 1

	if make_video:
		local_mp4_path_out = local_mp4_path[:-4] + '_trackedv2' + local_mp4_path[-4:]
		imageio.mimwrite(local_mp4_path_out, np.array(processed_video), fps=cap_fps)

	print('Run time is %s seconds' % (time.time() - start_time))
	return None #ideally would return number of tracked objects, confidence levels


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
	detection_confidence = params["confidence_threshold"]
	detection_model = params["yolo_model"]
	detection_implementation = params["yolo_implementation"]
	detection_frequency = 50

	track_objects(local_mp4_path=local_mp4_path,
				  detection_model=detection_model,detection_confidence=detection_confidence,
				  detection_implementation=detection_implementation,
				  detection_frequency=detection_frequency,tracking_model=tracking_model)