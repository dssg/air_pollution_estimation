from src.traffic_analysis.d04_modelling.trafficanalyserinterface import TrafficAnalyserInterface 
from src.traffic_analysis.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib, display_bboxes_on_frame, bbox_intersection_over_union
from src.traffic_analysis.d00_utils.video_helpers import write_mp4
from src.traffic_analysis.d00_utils.load_confs import load_parameters, load_paths
from src.traffic_analysis.d04_modelling.object_detection import detect_bboxes
from src.traffic_analysis.d04_modelling.trackinganalyser.vehiclefleet import VehicleFleet

import numpy as np
import pandas as pd 
import sys
import cv2
import time


class TrackingAnalyser(TrafficAnalyserInterface): 
    def __init__(self, video_dict, params, paths):
        """
        Detection and tracking arguments: 
        detection_model -- specify the name of model you want to use for detection 
        detection_implementation -- specify model to use for detection
        detection_frequency -- each detection_frequency num of frames, run obj detection alg again to detect new objs
        tracking_model -- specify name of model you want to use for tracking (currently only supports OpenCV trackers)
        iou_threshold -- specify threshold to use to decide whether two detected objs should be considered the same

        Stop start arguments:
        iou_convolution_window -- frame window size to perform iou computation on (to get an IOU time 
                                  series for each vehicle) 
        smoothing_method -- method to smooth the IOU time series for each vehicle
        stop_start_iou_threshold -- threshold to binarize the IOU time series into 0 or 1,denoting "moving" or "stopped"

        """
        super().__init__(video_dict, params, paths)
        self.detection_model = params['detection_model']
        self.detection_implementation = params['detection_implementation']
        self.tracking_model = params['opencv_tracker_type']
        self.iou_threshold = params['iou_threshold']
        self.detection_frequency = params['detection_frequency']


        self.iou_convolution_window = params['iou_convolution_window']
        self.smoothing_method = params['smoothing_method']
        self.stop_start_iou_threshold = params['stop_start_iou_threshold']


    def determine_new_bboxes(self, bboxes_tracked: list, bboxes_detected: list, iou_threshold: float = 0.1) -> list:
        """Return the indices of "new" bboxes in bboxes_detected so that a new tracker can be added for that 

        Keyword arguments: 

        bboxes_tracked: bboxes which are currently tracked. bboxes should be passed in in format (xmin,ymin,w,h)
        bboxes_detected: bboxes which are newly detected. bboxes should be passed in in format (xmin,ymin,w,h)
        iou_threshold: a detected bbox with iou below the iou_threshold (as compared to all existing, tracked bboxes) 
                       will be considered new. 
        """

        new_bboxes_inds = set(range(len(bboxes_detected)))  # init with all inds
        old_bboxes_inds = set()
        for i, box_a in enumerate(bboxes_detected):
            # if find a box which has high IOU with an already-tracked box, consider it an old box
            for box_b in bboxes_tracked:
                # format conversion needed
                iou = bbox_intersection_over_union(
                    bboxcv2_to_bboxcvlib(box_a), bboxcv2_to_bboxcvlib(box_b))
                if iou > iou_threshold:  # assume bbox has already been tracked and is not new
                    old_bboxes_inds.add(i)  # add to set

        new_bboxes_inds = list(new_bboxes_inds.difference(old_bboxes_inds))
        return new_bboxes_inds


    def track_objects(self, 
                      video: np.ndarray,
                      video_name: str,
                      video_time_length=10,
                      make_video=False, 
                      local_mp4_dir: str = None) -> VehicleFleet:
        """
        Given a path to an input video, this function will initialize a specified tracking algorithm 
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
        bboxes, labels, confs = detect_bboxes(frame=frame,
                                              model=detection_model,
                                              implementation=detection_implementation,
                                              selected_labels=True)
        # store info returned above in vehicleFleet object
        fleet = VehicleFleet(np.array(bboxes), 
                            np.array(labels), 
                            np.array(confs), 
                            video_name.replace(".mp4", ""))

        # Create MultiTracker object using bboxes, initialize multitracker
        multitracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            multitracker.add(create_tracker_by_name(
                tracking_model), frame, tuple(bbox))

        processed_video = []
        # Process video and track objects
        for frame_ind in range(n_frames):
            frame = video[frame_ind,:,:,:]
            # get updated location of objects in subsequent frames, update fleet obj
            success, bboxes_tracked = multitracker.update(frame)
            fleet.update_vehicles(bboxes_tracked)

            # draw tracked objects
            display_bboxes_on_frame(frame, bboxes_tracked, 
                                    fleet.compute_colors(), 
                                    fleet.compute_label_confs())

            # every x frames, re-detect boxes
            if frame_ind % detection_frequency == 0:
                # redetect bounding boxes
                bboxes_detected, labels_detected, confs_detected = detect_bboxes(frame=frame,
                                                                                 model=detection_model,
                                                                                 implementation=detection_implementation,
                                                                                 selected_labels=True)
                # re-initialize MultiTracker
                new_bbox_inds = self.determine_new_bboxes(bboxes_tracked,
                                                     bboxes_detected,
                                                     iou_threshold)
                new_bboxes = [bboxes_detected[i] for i in new_bbox_inds]
                new_labels = [labels_detected[i] for i in new_bbox_inds]
                new_confs = [confs_detected[i] for i in new_bbox_inds]

                # iterate through new bboxes
                for i, new_bbox in enumerate(new_bboxes):
                    multitracker.add(create_tracker_by_name(tracking_model),
                                     frame,
                                     tuple(new_bbox))

                # update fleet object
                if new_bboxes != []:
                    fleet.add_vehicles(np.array(new_bboxes),
                                       np.array(new_labels),
                                       np.array(new_confs))

            processed_video.append(frame)

            # code to display video frame by frame while it is bein processed
            # cv2.imshow('MultiTracker', frame)
            # # quit on ESC button
            # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            #   break

        if make_video:
            write_mp4(local_mp4_dir=local_mp4_dir,
                      mp4_name=video_name + "_tracked.mp4",
                      video=np.array(processed_video),
                      fps=video_frames_per_sec)
        print('Run time of tracking analyser for one video is %s seconds' % (time.time() - start_time))
        return fleet


    def construct_frame_level_df(self, video_dict) -> pd.DataFrame:
        frame_info_list = []
        for name,video in video_dict.items(): 
            fleet = self.track_objects(video,name)
            single_frame_level_df = fleet.report_frame_level_info()
            frame_info_list.append(single_frame_level_df)
        return pd.concat(frame_info_list)


    def reconstruct_fleet_from_single_frame_table(self,single_frame_level_df:pd.DataFrame) -> VehicleFleet:
        """Wrapper function for reconstructing a single fleet object from a frame_level_df. See VehicleFleet 
        class for more information.  
        """
        return VehicleFleet(frame_level_df = single_frame_level_df, load_from_pd = True)


    def construct_video_level_df(self,frame_level_df) -> pd.DataFrame: 
        """Construct video-level stats table using tracking techniques 

        Keyword arguments: 
        fleet -- instance of the VehicleFleet class, which stores tracking/obj detection info 
                 of all the vehicles in a video
        iou_convolution_window -- convolution window size when computing the iou between two bboxes
        smoothing_method -- method to use to smooth the iou time series; see stats_helpers.py
        stop_start_iou_threshold -- iou threshold between 0 and 1; above this, we consider the vehicle 
                                    as stopped; below, we consider the vehicle moving
        """
        video_info_list = []
        for _, single_frame_level_df in frame_level_df.groupby('cam_id'): 
            fleet = self.reconstruct_fleet_from_single_frame_table(single_frame_level_df)
            # compute the convolved IOU time series for each vehicle and smooth
            fleet.compute_iou_time_series(interval=self.iou_convolution_window)
            fleet.smooth_iou_time_series(smoothing_method=self.smoothing_method)
            # sample plotting options 
            # fleet.plot_iou_time_series(fig_dir="data", fig_name="param_tuning", smoothed=True)
            video_level_df = fleet.report_video_level_stats(fleet.compute_counts(), 
                                       *fleet.compute_stop_starts(self.stop_start_iou_threshold))
            video_info_list.append(video_level_df)
        return pd.concat(video_info_list)


if __name__ == '__main__':
    params = load_parameters()
    paths = load_paths()
    analyser = TrackingAnalyser(video_dict = {}, params, paths)
