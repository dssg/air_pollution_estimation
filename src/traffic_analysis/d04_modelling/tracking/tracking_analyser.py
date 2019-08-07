from traffic_analysis.d04_modelling.traffic_analyser_interface import TrafficAnalyserInterface
from traffic_analysis.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib, display_bboxes_on_frame, color_bboxes, \
    bbox_intersection_over_union
from traffic_analysis.d00_utils.video_helpers import write_mp4
from traffic_analysis.d04_modelling.tracking.vehicle_fleet import VehicleFleet
from traffic_analysis.d04_modelling.perform_detection_opencv import detect_objects_in_image as detect_objects_cv
from traffic_analysis.d04_modelling.perform_detection_tensorflow import detect_objects_in_image as detect_objects_tf
from traffic_analysis.d04_modelling.perform_detection_tensorflow import initialize_tensorflow_model

import numpy as np
import pandas as pd
import cv2
import time
import tensorflow as tf


class TrackingAnalyser(TrafficAnalyserInterface):
    def __init__(self, params, paths, s3_credentials):
        """
        Model-specific parameters initialized below: 

        (Object detection arguments:)
        detection_model -- specify the name of model you want to use for detection 
        detection_implementation -- specify model to use for detection
        detection_frequency -- each detection_frequency num of frames, run obj detection alg again to detect new objs
        tracking_model -- specify name of model you want to use for tracking (currently only supports OpenCV trackers)
        iou_threshold -- specify threshold to use to decide whether two detected objs should be considered the same
        detection_confidence_threshold -- conf above which to return label
        detection_nms_threshold -- yolo param
        selected_labels -- labels which we wish to detect

        (Stop start arguments:)
        iou_convolution_window -- frame window size to perform iou computation on (to get an IOU time 
                                  series for each vehicle) 
        smoothing_method -- method to smooth the IOU time series for each vehicle
        stop_start_iou_threshold -- threshold to binarize the IOU time series into 0 or 1,denoting "moving" or "stopped"
        """
        super().__init__(params, paths)
        self.detection_model = params['detection_model']
        self.detection_implementation = params['detection_implementation']
        self.tracker = self.create_tracker_by_name(tracker_type=params['opencv_tracker_type'])
        self.iou_threshold = params['iou_threshold']
        self.detection_frequency = params['detection_frequency']
        self.detection_confidence_threshold = params['detection_confidence_threshold']
        self.detection_nms_threshold = params['detection_nms_threshold']
        self.selected_labels = params['selected_labels']

        self.iou_convolution_window = params['iou_convolution_window']
        self.smoothing_method = params['smoothing_method']
        self.stop_start_iou_threshold = params['stop_start_iou_threshold']

        self.params = params
        self.paths = paths
        self.s3_credentials = s3_credentials

    def create_tracker_by_name(self, tracker_type: str):
        """Create tracker based on tracker name"""
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                         'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

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

    def determine_new_bboxes(self, bboxes_tracked: list, bboxes_detected: list) -> list:
        """Return the indices of "new" bboxes in bboxes_detected so that a new tracker can be added for that 

        Keyword arguments: 

        bboxes_tracked: bboxes which are currently tracked. bboxes should be passed in in format (xmin,ymin,w,h)
        bboxes_detected: bboxes which are newly detected. bboxes should be passed in in format (xmin,ymin,w,h)
        iou_threshold: a detected bbox with iou below the iou_threshold (as compared to all existing, tracked bboxes) 
                       will be considered new. 
        """

        new_bboxes_inds = set(range(len(bboxes_detected))
                              )  # init with all inds
        old_bboxes_inds = set()
        for i, box_a in enumerate(bboxes_detected):
            # if find a box which has high IOU with an already-tracked box, consider it an old box
            for box_b in bboxes_tracked:
                # format conversion needed
                iou = bbox_intersection_over_union(
                    bboxcv2_to_bboxcvlib(box_a), bboxcv2_to_bboxcvlib(box_b))
                if iou > self.iou_threshold:  # assume bbox has already been tracked and is not new
                    old_bboxes_inds.add(i)  # add to set

        new_bboxes_inds = list(new_bboxes_inds.difference(old_bboxes_inds))
        return new_bboxes_inds

    def detect_and_track_objects(self,
                                 video: np.ndarray,
                                 video_name: str,
                                 video_time_length=10,
                                 make_video=False,
                                 local_mp4_dir: str = None) -> VehicleFleet:
        """Code to track
        This function will initialize a specified tracking algorithm 
        (currently only supports OpenCV's built in multitracker methods) with the specified object 
        detection algorithm. Each detection_frequency frames, the object detection will be run again 
        to detect any new objects which have entered the frame. A VehicleFleet object is used to track the 
        initial detection confidence and label for each vehicle as it is detected, and the updated locations 
        of the bounding boxes for each vehicle each frame. The VehicleFleet object also performs IOU computations
        on the stored bounding box information to get counts and stop starts. 

        Keyword arguments:

        video -- np array in format (frame_count,frame_height,frame_width,3)
        video_name -- name of video to run on (include .mp4 extension) 
        video_time_length -- specify length of video 
        make_video -- if true, will write video to local_mp4_dir with name local_mp4_name_tracked.mp4
        local_mp4_dir -- path to directory to store video in 
        """

        start_time = time.time()
        # Create a video capture object to read videos
        n_frames = video.shape[0]

        # assumes vid_length in seconds
        video_frames_per_sec = int(n_frames / video_time_length)

        # initialize bboxes on first frame using a detection alg
        first_frame = video[0, :, :, :]

        if self.detection_model == 'yolov3' or self.detection_model == 'yolov3-tiny':
            bboxes, labels, confs = detect_objects_cv(image_capture=first_frame,
                                                      params=self.params,
                                                      paths=self.paths,
                                                      s3_credentials=self.s3_credentials,
                                                      selected_labels=self.selected_labels)
        elif self.detection_model == 'yolov3_tf':
            sess = tf.Session()
            model_initializer, init_data, detection_model = initialize_tensorflow_model(params=self.params,
                                                                                        paths=self.paths,
                                                                                        s3_credentials=self.s3_credentials,
                                                                                        sess=sess)

            bboxes, labels, confs = detect_objects_tf(image_capture=first_frame,
                                                      paths=self.paths,
                                                      detection_model=detection_model,
                                                      model_initializer=model_initializer,
                                                      init_data=init_data,
                                                      sess=sess,
                                                      selected_labels=self.selected_labels)

        # store info returned above in vehicleFleet object
        fleet = VehicleFleet(bboxes=np.array(bboxes),
                             labels=np.array(labels),
                             confs=np.array(confs),
                             video_name=video_name.replace(".mp4", ""))

        # Create MultiTracker object using bboxes, initialize multitracker
        multi_tracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            multi_tracker.add(newTracker=self.tracker,
                              image=first_frame,
                              boundingBox=tuple(bbox))

        if make_video:
            processed_video = []

        # Process video and track objects
        for frame_ind in range(1, n_frames):
            frame = video[frame_ind, :, :, :]
            # get updated location of objects in subsequent frames, update fleet obj
            success, bboxes_tracked = multi_tracker.update(image=frame)
            fleet.update_vehicles(np.array(bboxes_tracked))

            if make_video:
                # draw tracked objects
                display_bboxes_on_frame(frame, bboxes_tracked,
                                        color_bboxes(fleet.labels),
                                        fleet.compute_label_confs())

            # every x frames, re-detect boxes
            if frame_ind % self.detection_frequency == 0:
                # redetect bounding boxes
                if self.detection_model == 'yolov3' or self.detection_model == 'yolov3-tiny':
                    bboxes_detected, labels_detected, confs_detected = detect_objects_cv(image_capture=first_frame,
                                                                                         params=self.params,
                                                                                         paths=self.paths,
                                                                                         s3_credentials=self.s3_credentials,
                                                                                         selected_labels=self.selected_labels)
                elif self.detection_model == 'yolov3_tf':
                    bboxes_detected, labels_detected, confs_detected = detect_objects_tf(image_capture=frame,
                                                                                         paths=self.paths,
                                                                                         detection_model=detection_model,
                                                                                         model_initializer=model_initializer,
                                                                                         init_data=init_data,
                                                                                         sess=sess,
                                                                                         selected_labels=self.selected_labels)

                # re-initialize MultiTracker
                new_bbox_inds = self.determine_new_bboxes(bboxes_tracked,
                                                          bboxes_detected)

                # update fleet object
                if len(new_bbox_inds) > 0:
                    new_bboxes = [bboxes_detected[i] for i in new_bbox_inds]
                    new_labels = [labels_detected[i] for i in new_bbox_inds]
                    new_confs = [confs_detected[i] for i in new_bbox_inds]

                    fleet.add_vehicles(np.array(new_bboxes),
                                       np.array(new_labels),
                                       np.array(new_confs))

                    # iterate through new bboxes
                    for i, new_bbox in enumerate(new_bboxes):
                        multi_tracker.add(newTracker=self.tracker,
                                          image=frame,
                                          boundingBox=tuple(new_bbox))

            if make_video:
                processed_video.append(frame)

                # code to display video frame by frame while it is being processed
                # cv2.imshow('MultiTracker', frame)
                # # quit on ESC button
                # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                #   break
        if make_video:
            write_mp4(local_mp4_dir=local_mp4_dir,
                      mp4_name=video_name + "_tracked.mp4",
                      video=np.array(processed_video),
                      fps=video_frames_per_sec)

        print('Run time of tracking analyser for one video is %s seconds' %
              (time.time() - start_time))

        if self.detection_model == 'yolov3_tf':
            sess.close()

        return fleet

    def construct_frame_level_df(self, video_dict) -> pd.DataFrame:
        """Construct frame level df for multiple videos 
        """

        # Check that video doesn't come from in-use camera (some are)
        for video_name in list(video_dict.keys()):
            n_frames = video_dict[video_name].shape[0]
            if n_frames < 75:
                del video_dict[video_name]
                print("Video ", video_name,
                      " has been removed from processing because it may be invalid")

        frame_info_list = []
        for video_name, video in video_dict.items():
            fleet = self.detect_and_track_objects(video, video_name)
            single_frame_level_df = fleet.report_frame_level_info()
            frame_info_list.append(single_frame_level_df)
        return pd.concat(frame_info_list)

    def construct_video_level_df(self, frame_level_df) -> pd.DataFrame:
        """Construct video-level stats table using tracking techniques 

        Keyword arguments: 

        frame_level_df -- df returned by above function
        """
        if frame_level_df.empty:
            return frame_level_df

        video_info_list = []
        for _, single_frame_level_df in frame_level_df.groupby(['camera_id', 'video_upload_datetime']):
            fleet = VehicleFleet(frame_level_df=single_frame_level_df, load_from_pd=True)
            # compute the convolved IOU time series for each vehicle and smooth
            fleet.compute_iou_time_series(interval=self.iou_convolution_window)
            fleet.smooth_iou_time_series(smoothing_method=self.smoothing_method)
            # sample plotting options
            # fleet.plot_iou_time_series(fig_dir="data", fig_name="param_tuning", smoothed=True)
            video_level_df = fleet.report_video_level_stats(fleet.compute_counts(),
                                                            *fleet.compute_stop_starts(self.stop_start_iou_threshold))
            video_info_list.append(video_level_df)
        return pd.concat(video_info_list)
