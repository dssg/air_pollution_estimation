from traffic_analysis.d00_utils.bbox_helpers import color_bboxes, bbox_intersection_over_union, bboxcv2_to_bboxcvlib
from traffic_analysis.d00_utils.stats_helpers import time_series_smoother
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
import os
import numpy as np
import pandas as pd
import math
import collections
import matplotlib.pyplot as plt
import datetime


class VehicleFleet():
    """
    The purpose of the vehicleFleet object is to store information about each vehicle detected/tracked in 
    a video. This information includes the bounding box location with each frame, the vehicle labels, 
    and the initial detection confidences. Throughout this class, we use numpy arrays, following the 
    convention that axis 0 corresponds to vehicle index, axis 1 corresponds to the information for each 
    vehicle we are interested in, and axis 2 corresponds to the frame/time dimension for the info 
    recorded by axis 1 (if relevant). 
    """

    def __init__(self, bboxes: np.ndarray = None,
                       labels: np.ndarray = None,
                       confs: np.ndarray = None,
                       video_name: str = None,
                       frame_level_df: pd.DataFrame = None,
                       load_from_pd = False):
        """Initialize the vehicleFleet object, either from a saved dataframe, or from the info returned by
        running object detection algs on the first frame of a video.

        Keyword arguments: 

        load_from_pd -- if this is True, will reconstruct VehicleFleet object from a frame-level stats
                        table. Else constructs it from scratch using bbox, label, confs, and video name
        frame_level_df -- a pandas df corresponding to one video, which follows the schema for a
                          frame-level stats df
        bboxes -- pass in using cv2 format (xmin, ymin, width, height). Each vehicle should be axis 0,
                  bounding boxes should be axis 1. 
        labels -- label assigned to detected objects 
        confs -- confidence returned by detection alg 
        video_name -- name of the video file from s3 bucket
        """
        # bool for tracking whether there is a fake head vehicle
        # used to handle case if no vehicles detected in video
        self.fake_head_vehicle = False

        if load_from_pd:
            # sort to ensure that the ordering of self.labels and self.confs corresps to vehicle id
            frame_level_df = frame_level_df.sort_values(by=['camera_id', 'frame_id'], axis='index')
            self.labels = np.array(frame_level_df[frame_level_df["frame_id"] == 0]["vehicle_type"])
            self.confs = np.array(frame_level_df[frame_level_df["frame_id"] == 0]["confidence"])
            # get static info
            self.camera_id = frame_level_df["camera_id"].iloc[0]
            self.video_upload_datetime = frame_level_df["video_upload_datetime"].iloc[0]

            # extract bbox info into np array
            frame_level_df['bboxes'] = frame_level_df['bboxes'].apply(np.array)
            bboxes_np = np.array(frame_level_df[['frame_id', 'bboxes']].groupby('frame_id')['bboxes'].apply(np.vstack))
            num_vehicles, num_frames = bboxes_np[0].shape[0], bboxes_np.shape[0]

            # reshape array
            self.bboxes = np.zeros((num_vehicles, 4, num_frames))
            for i in range(num_frames):
                self.bboxes[:,:,i] = bboxes_np[i]
        else:
            # check if the bboxes are empty
            if bboxes.size == 0: 
                # add fake vehicle to ensure that self.bboxes.shape[2] == number frames in video
                bboxes = np.zeros((1,4))
                labels = np.array(["fake_head_vehicle"])
                confs = np.array([0.0])
                # update fake head status bool
                self.fake_head_vehicle = True 

            assert bboxes.shape[1] == 4
            # add a dimension for time
            self.bboxes = np.expand_dims(bboxes, axis=2)
            self.labels = labels
            self.confs = confs
            self.camera_id, self.video_upload_datetime = parse_video_or_annotation_name(video_name)

    def add_vehicles(self, new_bboxes: np.ndarray, new_labels: np.ndarray, new_confs: np.ndarray):
        """Adds new vehicles to the vehicleFleet, creating appropriate bbox location "history" for the 
        self.bboxes numpy array 

        Keyword arguments should be in same format as for the init
        """
        current_time_t = self.bboxes.shape[2]
        num_new_vehicles = new_bboxes.shape[0]
        # create bboxes of all zeros to denote that the vehicle didn't exist at previous times
        new_vehicle_history = np.zeros((num_new_vehicles, 4, current_time_t-1))
        new_bboxes = np.concatenate((np.expand_dims(new_bboxes, axis=2),
                                     new_vehicle_history), axis=2)

        self.bboxes = np.concatenate((self.bboxes, new_bboxes), axis=0)
        self.labels = np.concatenate((self.labels, new_labels), axis=0)
        self.confs = np.concatenate((self.confs, new_confs), axis=0)
        return

    def update_vehicles(self, bboxes_time_t: np.ndarray):
        """Updates the bbox location for current vehicles by appending to self.bboxes in the 
        time axis for all existing vehicles. 

        Keyword arguments: 

        bboxes_time_t -- updated location for bboxes for all existing vehicles. pass in using cv2 
                       format (xmin, ymin, width, height). Each vehicle should be axis 0, 
                       bounding boxes should be axis 1.
        """
        # if no vehicles tracked, or all tracked objects have exited frames
        if bboxes_time_t.size == 0: 
            # create bboxes of 0s to append to self.bboxes to ensure 
            # that each frame in a video corresponds to a subarray in self.bboxes
            num_vehicles = self.bboxes.shape[0]
            bboxes_time_t = np.zeros((num_vehicles,4))

        else: # check tracking format is correct 
            assert bboxes_time_t.shape[1] == 4

        self.bboxes = np.concatenate((self.bboxes, np.expand_dims(bboxes_time_t, axis=2)), 
                                        axis=2)
        return

    def compute_counts(self) -> dict:
        """Get counts of each vehicle type 
        """
        count = collections.Counter(self.labels)
        if self.fake_head_vehicle:
            count -= 1
        return count

    def compute_iou_time_series(self, interval: int = 15):
        """Compute a convolved IOU time series for each vehicle

        Keyword arguments: 

        interval -- convolution window size. Compute IOU between frames that 
                    are separated by this window size. 
        """
        self.iou_interval = interval

        num_vehicles, _, num_frames = self.bboxes.shape
        iou_time_series = np.zeros((num_vehicles, num_frames - interval))

        # compare bboxes at timepoints t0,t1 for each vehicle; compute iou
        # t0, t1 are separated by the interval parameter
        for i in range(0, num_frames-interval):
            bboxes_time_t0 = self.bboxes[:, :, i]
            bboxes_time_t1 = self.bboxes[:, :, i+interval]

            for j in range(num_vehicles):
                iou_time_series[j, i] = bbox_intersection_over_union(bboxcv2_to_bboxcvlib(bboxes_time_t0[j]),
                                                                     bboxcv2_to_bboxcvlib(bboxes_time_t1[j]))
        self.iou_time_series = iou_time_series
        # if self.fake_head_vehicle: # handle fake head vehicle
            # if num_vehicles == 1: # no real vehicles detected

            # self.iou_time_series = self.iou_time_series[1:, :, :]
        return 

    def smooth_iou_time_series(self, smoothing_method: str, **smoothing_settings):
        """Wrapper function for smoothing the iou time series

        Keyword arguments 

        smoothing_method -- e.g. "moving_avg" (also this method is the best performing)
        smoothing_settings -- see the time_series_smoother function; various settings for the 
                             smoothing methods are available. Pass in whatever keyword 
                             parameters are desired.
        """
        # some good settings for various smoothing methods
        default_settings = {"window_size": 25}
        for setting_name, setting_value in smoothing_settings.items():
            default_settings[setting_name] = setting_value
        self.smoothed_iou_time_series = time_series_smoother(self.iou_time_series,
                                                             method=smoothing_method,
                                                             **default_settings)
        return

    def compute_stop_starts(self, stop_start_iou_threshold: float = 0.85, from_smoothed=True) -> dict:
        """ Compute the stop starts by thresholding the IOU time series data. Performance is best 
        when using the smoothed IOU time series. 

        Keyword arguments:

        stop_start_iou_threshold -- above this threshold, IOU is rounded to 1. Under this threshold, IOU is rounded 
                                    to 0. 1 indicates that the vehicle is in motion, 0 indicates it is stopped. 
        from_smoothed -- Whether or not to use the smoothed IOU time series data in computing stop starts.
        """
        motion_array = np.copy(
            self.smoothed_iou_time_series) if from_smoothed else np.copy(self.iou_time_series)

        # don't need to worry about fake head vehicle bc iou_time_series for
        # this vehicle should always be 0 
        if self.fake_head_vehicle: 
            assert np.sum(motion_array[0,:]) == 0

        # round iou values to binary values
        motion_array[motion_array > stop_start_iou_threshold] = 1
        motion_array[motion_array <= stop_start_iou_threshold] = 0

        # iterate over vehicles, frames axes of the motion_array
        num_vehicles, num_frames = motion_array.shape
        stop_counter, start_counter = [], []
        for vehicle_idx in range(num_vehicles):
            # initialize the "motion status" by looking at first iou value
            motion_status_prev = motion_array[vehicle_idx, 0]

            for frame_idx in range(num_frames):
                motion_status_current = motion_array[vehicle_idx, frame_idx]
                # TODO: get stops, starts, by vehicle types, get
                # change in motion status
                if motion_status_current != motion_status_prev:
                    if motion_status_prev == 0:
                        # get the type of the object that just stopped
                        stop_counter.append(self.labels[vehicle_idx])
                    elif motion_status_prev == 1:
                        start_counter.append(self.labels[vehicle_idx])

                    motion_status_prev = motion_status_current

        return collections.Counter(stop_counter), collections.Counter(start_counter)

    def compute_label_confs(self):
        """Append label, id, and confidence for each vehicle for plotting purposes
        """
        label_confs = [label + ', id=' + str(i) + ', ' + str(format(
            self.confs[i] * 100, '.2f')) + '%' for i, label in enumerate(self.labels)]
        return label_confs

    def plot_iou_time_series(self, fig_dir: str, fig_name: str, smoothed=False, vehicle_ids=None):
        """Visualize the iou_time_series as a multi-line graph

        fig_dir -- path to dir save plot to
        fig_name -- desired name of the figure; do NOT include file extension
        smoothed -- if true, plot the smoothed iou time series
        vehicle_ids -- if specified, only visualize the iou time series for these vehicles
        """
        iou = self.smoothed_iou_time_series if smoothed else self.iou_time_series

        num_vehicles, num_ious = iou.shape[0], iou.shape[1]

        if vehicle_ids is None:
            vehicles_ids = range(num_vehicles)

        # plot each vehicle
        vehicle_colors = np.array(color_bboxes(self.labels)) / 255
        iou_inds = np.arange(num_ious)
        for i in range(num_vehicles):
            iou_vehicle = iou[i, :]
            # catch na's or infs
            mask1, mask2 = np.isnan(iou_vehicle), np.isfinite(iou_vehicle)
            label = self.labels[i+1] if self.fake_head_vehicle else self.labels[i]
            plt.plot(iou_inds[~mask1 & mask2], iou_vehicle[~mask1 & mask2],
                     label="vehicle " + str(i) + "; type " + label)
        plt.legend(loc='lower right')
        plt.xlabel("IOU over all frames in video, interval = " +
                   str(self.iou_interval))
        plt.savefig(os.path.join(fig_dir, fig_name + ".pdf"))
        plt.close()

    def report_frame_level_info(self) -> pd.DataFrame:
        """Converts the information stored in the VehicleFleet class to a frame level pd dataframe.
        Fake head vehicles are removed here, so that report_video_level_info doesn't have to handle
        this.
        """
        column_names = ['camera_id', 'video_upload_datetime',
                        'frame_id', 'vehicle_id', 'vehicle_type', 
                        'confidence', 'bboxes']
        if self.fake_head_vehicle: # handle fake head before reporting 
            if self.bboxes.shape[0] == 1: #only a fake head vehicle, no other vehicles in fleet 
                return pd.DataFrame([[self.camera_id, self.video_upload_datetime, 
                                      math.nan, math.nan, math.nan, 
                                      math.nan, math.nan]], columns = column_names)
            else: 
                self.bboxes = self.bboxes[1,:,:] # remove fake head vehicle 
                self.labels = self.bboxes[1:]
                self.confs = self.bboxes[1:]
                self.fake_head_vehicle = False

        num_vehicles, _, num_frames = self.bboxes.shape

        # add vehicle index for each frame to all frames at once
        vehicle_ids = np.arange(0, num_vehicles)
        add_vehicle_ids = lambda all_frames, vehicle_ids: np.concatenate(
                            [all_frames,np.tile(vehicle_ids, (num_frames, 1, 1)).transpose()],
                            axis = 1)

        # add the frame id to each frame
        frame_info_all = add_vehicle_ids(self.bboxes,vehicle_ids)
        add_frame_id = lambda single_frame, frame_id: np.concatenate([single_frame, (np.ones(num_vehicles)*frame_id)[:,np.newaxis]], axis = 1)
        # convert 3d array into 2d pd dataframe
        frame_info_list = [add_frame_id(frame_info_all[:,:,frame_id],frame_id) for frame_id in range(num_frames)]
        stacked = np.vstack(frame_info_list)
        frame_level_info_df = pd.DataFrame(stacked)

        frame_level_info_df["bboxes"] = frame_level_info_df.iloc[:,:4].values.tolist()
        # drop the numeric columns used above, rename the rest
        frame_level_info_df = frame_level_info_df.drop(columns = [0,1,2,3]).rename({4:"vehicle_id",
                                                                                    5:"frame_id"},
                                                                                    axis = "columns")
        # column type conversions
        frame_level_info_df["vehicle_id"] = frame_level_info_df["vehicle_id"].astype('int')
        frame_level_info_df["frame_id"] = frame_level_info_df["frame_id"].astype('int')

        # add vehicle types and confidences
        frame_level_info_df["vehicle_type"] = frame_level_info_df.apply(lambda row: self.labels[row['vehicle_id']], axis = 1)
        frame_level_info_df["confidence"] = frame_level_info_df.apply(lambda row: self.confs[row['vehicle_id']], axis = 1)
        frame_level_info_df["video_upload_datetime"] = self.video_upload_datetime
        frame_level_info_df["camera_id"] = self.camera_id
        # TODO: change vehicle_type to vehicle_type_id, camera_id to proper integer camera_id
        # frame_level_info_df["camera_id"] = frame_level_info_df["camera_id"].astype('int')

        # reorder columns
        frame_level_info_df = frame_level_info_df[column_names]

        return frame_level_info_df

    def report_video_level_stats(self, vehicle_counts: dict,
                               vehicle_stop_counts: dict,
                               vehicle_start_counts: dict) -> pd.DataFrame:
        """ Combine the counting dictionaries of vehicle stops, starts, and counts into
        one nice pandas dataframe. 

        Keyword arguments

        vehicle_counts -- dict with keys as vehicle types, values as vehicle counts
        vehicle_stop_counts -- dict with keys as vehicle types, values as vehicle stops
        vehicle_start_counts -- dict with keys as vehicle types, values as vehicle starts
        """
        column_names = ['camera_id', 'video_upload_datetime',
                        'vehicle_type', 'counts', 'stops', 'starts']

        counts_df = pd.DataFrame.from_dict(vehicle_counts,
                                          orient='index', columns=['counts'],)
        stops_df = pd.DataFrame.from_dict(vehicle_stop_counts,
                                          orient='index', columns=['stops'])
        starts_df = pd.DataFrame.from_dict(vehicle_start_counts,
                                           orient='index', columns=['starts'])

        # combine into one dataframe
        video_level_stats_df = counts_df.join([stops_df, starts_df], how='outer',
                                 sort=True).fillna(0)
        video_level_stats_df["camera_id"] = self.camera_id
        video_level_stats_df["video_upload_datetime"] = self.video_upload_datetime
        # rownames to a column
        video_level_stats_df.index.name = 'vehicle_type'
        video_level_stats_df.reset_index(inplace=True)
        # reorder columns
        video_level_stats_df = video_level_stats_df[column_names]
        return video_level_stats_df
