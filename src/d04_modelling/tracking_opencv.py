from src.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib, display_bboxes_on_frame,bbox_intersection_over_union
from src.d00_utils.video_helpers import write_mp4
from src.d04_modelling.object_detection import detect_bboxes
from src.d04_modelling.vehiclefleet import VehicleFleet
import numpy as np
import sys, os
import cv2
import yaml
import time
import pickle as pkl


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


def determine_new_bboxes(bboxes_tracked:list, bboxes_detected:list, iou_threshold:float = 0.1) -> list:
	"""Return the indices of "new" bboxes in bboxes_detected so that a new tracker can be added for that 
	
	Keyword arguments: 

	bboxes_tracked: bboxes which are currently tracked. bboxes should be passed in in format (xmin,ymin,w,h)
	bboxes_detected: bboxes which are newly detected. bboxes should be passed in in format (xmin,ymin,w,h)
	iou_threshold: a detected bbox with iou below the iou_threshold (as compared to all existing, tracked bboxes) 
				   will be considered new. 
	"""

	new_bboxes_inds = set(range(len(bboxes_detected))) #init with all inds
	old_bboxes_inds = set()
	for i,box_a in enumerate(bboxes_detected):
		# if find a box which has high IOU with an already-tracked box, consider it an old box
		for box_b in bboxes_tracked:
			# format conversion needed
			iou = bbox_intersection_over_union(bboxcv2_to_bboxcvlib(box_a), bboxcv2_to_bboxcvlib(box_b))
			if iou > iou_threshold: #assume bbox has already been tracked and is not new 
				old_bboxes_inds.add(i) #add to set

	new_bboxes_inds=list(new_bboxes_inds.difference(old_bboxes_inds))
	return new_bboxes_inds


def track_objects(local_mp4_dir:str,
				  local_mp4_name:str,
				  detection_model:str,
				  detection_confidence:float,
				  detection_implementation:str,
				  detection_frequency:int,
				  tracking_model:str,
				  iou_threshold:float,
				  selected_labels=True,
				  video_time_length=10,
				  make_video=True) -> (list,list,dict):
	"""
	Given a path to an input video, this function will initialize a specified tracking algorithm 
	(currently only supports OpenCV's built in multitracker methods) with the specified object 
	detection algorithm. Each detection_frequency frames, the object detection will be run again 
	to detect any new objects which have entered the frame. Returns labels of detected objects, the
	detection confidence, and the counts. 

	Keyword arguments

	local_mp4_dir -- path to directory to store video in 
	local_mp4_name -- name of video to run on (include .mp4 extension) 
	detection_model -- specify the name of model you want to use for detection 
	detection_confidence -- objects detected with lower than this level of confidence will NOT be tracked
	detection_implementation -- 
	detection_frequency -- each detection_frequency num of frames, run obj detection alg again to detect new objs
	tracking_model -- specify name of model you want to use for tracking (currently only supports OpenCV trackers)
	iou_threshold -- specify threshold to use 
	selected_labels -- if a list of labels is supplied, only bboxes with these labels will be tracked
	video_time_length -- specify length of video 
	make_video -- if true, will write video to local_mp4_dir with name local_mp4_name_tracked.mp4
	"""

	start_time = time.time()
	
	# Create a video vid_objture object to read videos
	vid_obj = cv2.VideoCapture(local_mp4_dir+ "/" + local_mp4_name)
	n_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
	vid_obj_frames_per_sec = int(n_frames / video_time_length)  # assumes vid_length in seconds

	# Read first frame
	success, frame = vid_obj.read()
	if not success:
		print('Failed to read video')
		sys.exit(0)

	bboxes, labels, confs = detect_bboxes(frame=frame,
										model=detection_model,
										# confidence=detection_confidence,
										implementation=detection_implementation,
										selected_labels = selected_labels)
	fleet = VehicleFleet(np.array(bboxes), np.array(labels), np.array(confs))
	# Create MultiTracker object
	multitracker = cv2.MultiTracker_create()

	# Initialize MultiTracker
	for bbox in bboxes:
		multitracker.add(create_tracker_by_name(tracking_model), frame, bbox)

	processed_video = []
	frame_counter = 0
	# Process video and track objects
	while vid_obj.isOpened():
		success, frame = vid_obj.read()
		if not success: break

		# get updated location of objects in subsequent frames
		success, bboxes_tracked = multitracker.update(frame)
		fleet.update_vehicles(bboxes_tracked)

		# draw tracked objects
		display_bboxes_on_frame(frame, bboxes_tracked, fleet.compute_colors(), fleet.compute_label_confs())

		# every x frames, re-detect boxes
		if frame_counter%detection_frequency == 0: 
			# redetect bounding boxes
			bboxes_detected, labels_detected, \
			confs_detected = detect_bboxes(frame=frame,
											model=detection_model,
											# confidence=detection_confidence,
											implementation=detection_implementation,
											selected_labels = selected_labels)
			# re-initialize MultiTracker
			new_bbox_inds = determine_new_bboxes(bboxes_tracked, bboxes_detected, iou_threshold)
			new_bboxes, new_labels, new_confs= [bboxes_detected[i] for i in new_bbox_inds],\
											   [labels_detected[i] for i in new_bbox_inds],\
											   [confs_detected[i] for i in new_bbox_inds]

			# iterate through new bboxes
			for i,new_bbox in enumerate(new_bboxes):
				multitracker.add(create_tracker_by_name(tracking_model), frame, new_bbox)

			#update fleet object
			if new_bboxes!=[]:
				fleet.add_vehicles(np.array(new_bboxes),np.array(new_labels),np.array(new_confs))

		processed_video.append(frame)
		frame_counter += 1

		# code to display video frame by frame while it is bein processed
		# cv2.imshow('MultiTracker', frame)
		# # quit on ESC button
		# if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
		# 	break

	if make_video:
		write_mp4(local_mp4_dir=local_mp4_dir,
							mp4_name=local_mp4_name[:-4] + "_tracked.mp4", 
							video=np.array(processed_video),
							fps=vid_obj_frames_per_sec)

	print('Run time is %s seconds' % (time.time() - start_time))
	# return fleet.labels, fleet.confs, fleet.compute_counts()
	return fleet


if __name__ == '__main__':
	#config stuff
	basepath=os.path.dirname(__file__) #path of current script

	os.chdir(".")
	with open('conf/base/parameters.yml') as f:
	   params = yaml.safe_load(f)['modelling']

	#get a video from local
	local_mp4_dir = os.path.abspath(os.path.join(basepath,"..", "..","data/sample_video_data"))
	local_mp4_name = "testvid.mp4"
	tracking_model = params["opencv_tracker_type"]
	detection_confidence = params["detection_confidence_threshold"]
	detection_model = params["yolo_model"]
	detection_implementation = params["yolo_implementation"]
	detection_frequency = params["detection_frequency"]
	iou_threshold = params["iou_threshold"]

	# fleet = track_objects(local_mp4_dir=local_mp4_dir, local_mp4_name=local_mp4_name,
	# 			  detection_model=detection_model,detection_confidence=detection_confidence,
	# 			  detection_implementation=detection_implementation,
	# 			  detection_frequency=detection_frequency,tracking_model=tracking_model,
	# 			  # selected_labels=selected_labels,
	# 			  iou_threshold=iou_threshold, make_video=True)

	# pkl_out = open("fleet_obj.pkl", "wb")
	# pkl.dump(fleet, pkl_out)
	# pkl_out.close()


	pkl_in = open("fleet_obj.pkl", "rb")
	fleet = pkl.load(pkl_in)
	# print(fleet.labels,fleet.confs,fleet.compute_counts())
	fleet.compute_iou_all()
	fleet.plot_iou_all()
