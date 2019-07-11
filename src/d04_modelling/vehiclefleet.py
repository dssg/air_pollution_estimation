from src.d00_utils.bbox_helpers import color_bboxes, bbox_intersection_over_union, vectorized_intersection_over_union, bboxcv2_to_bboxcvlib
from src.d00_utils.stats_helpers import time_series_smoother
import os
import numpy as np
import pandas as pd 
import collections
import matplotlib.pyplot as plt


class VehicleFleet():
    """
    The purpose of the vehicleFleet object is to store information about each vehicle detected/tracked in 
    a video. This information includes the bounding box location with each frame, the vehicle labels, 
    and the initial detection confidences. Throughout this class, we use numpy arrays, following the 
    convention that axis 0 corresponds to vehicle index, axis 1 corresponds to the information for each 
    vehicle we are interested in, and axis 2 corresponds to the frame/time dimension for the info 
    recorded by axis 1 (if relevant). 
    """

    def __init__(self, bboxes: np.ndarray, labels: np.ndarray, confs: np.ndarray, video_name:str):
        """Initialize the vehicleFleet with the bounding boxes for one set of vehicles, as 
        detected from one frame. 

        Keyword arguments: 

        bboxes -- pass in using cv2 format (xmin, ymin, width, height). Each vehicle should be axis 0, 
                  bounding boxes should be axis 1. 
        labels -- label assigned to detected objects 
        confs -- confidence returned by detection alg 
        video_name 
        """
        assert bboxes.shape[1] == 4

        # add a dimension for time
        self.bboxes = np.expand_dims(bboxes, axis=2)
        self.labels = labels
        self.confs = confs
        # TODO: implement a name parser to get the camera id and time from the video name
        self.video_name = video_name

    def add_vehicles(self, new_bboxes: np.ndarray, new_labels: np.ndarray, new_confs: np.ndarray):
        """Adds new vehicles to the vehicleFleet, creating appropriate bbox location "history" for the 
        self.bboxes numpy array 

        job: append more vehicles, come up with na arrays for prev history

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
        assert bboxes_time_t.shape[1] == 4
        self.bboxes = np.concatenate(
            (self.bboxes, np.expand_dims(bboxes_time_t, axis=2)), axis=2)
        return

    def compute_label_confs(self):
        """Append label, id, and confidence for each vehicle for plotting purposes
        """
        label_confs = [label + ', id=' + str(i) + ', ' + str(format(
            self.confs[i] * 100, '.2f')) + '%' for i, label in enumerate(self.labels)]
        return label_confs

    def compute_colors(self) -> list:
        """Assigns a color to each label currently present by vehicleFleet
        """
        return color_bboxes(self.labels)

    def compute_counts(self) -> dict:
        """Get counts of each vehicle type 
        """
        return collections.Counter(self.labels)

    def compute_iou_time_series(self, interval: int = 15):
        """Compute a convolved IOU time series for each vehicle

        Keyword arguments: 

        interval -- convolution window size. Compute IOU between frames that 
                    are separated by this window size. 
        """
        self.iou_interval = interval

        num_frames = self.bboxes.shape[2]
        num_vehicles = self.bboxes.shape[0]
        iou_time_series = np.zeros((num_vehicles, num_frames - interval))

        # compare bboxes at timepoints t0,t1 for each vehicle; compute iou
        # t0, t1 are separated by the interval parameter
        for i in range(0, num_frames-interval):
            bboxes_time_t0 = self.bboxes[:, :, i]
            bboxes_time_t1 = self.bboxes[:, :, i+interval]

            for j in range(num_vehicles):

                iou_time_series[j, i] = bbox_intersection_over_union(bboxcv2_to_bboxcvlib(bboxes_time_t0[j]),
                                                                     bboxcv2_to_bboxcvlib(bboxes_time_t1[j]))

            # iou_time_series[:,i] = vectorized_intersection_over_union(bboxcv2_to_bboxcvlib(bboxes_time_t0, vectorized = True),
                # bboxcv2_to_bboxcvlib(bboxes_time_t1, vectorized = True))
            # assert iou_time_series_ind == iou_time_series[0,i], str(iou_time_series_ind) + str(iou_time_series[0,i])
        self.iou_time_series = iou_time_series
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
        default_settings = {"window_size":25} 
        for setting_name, setting_value in smoothing_settings.items(): 
            default_settings[setting_name] = setting_value
        self.smoothed_iou_time_series = time_series_smoother(self.iou_time_series,
                                                             method=smoothing_method,
                                                             **default_settings)
        return

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
        vehicle_colors = np.array(self.compute_colors()) / 255
        iou_inds = np.arange(num_ious)
        for i in range(num_vehicles):
            iou_vehicle = iou[i, :]
            # catch na's or infs
            mask1, mask2 = np.isnan(iou_vehicle), np.isfinite(iou_vehicle)
            plt.plot(iou_inds[~mask1 & mask2], iou_vehicle[~mask1 & mask2],
                     # color = vehicle_colors[i], #color line chart by vehicle type
                     label="vehicle " + str(i) + "; type " + self.labels[i])
        plt.legend(loc='lower right')
        plt.xlabel("IOU over all frames in video, interval = " +
                   str(self.iou_interval))
        plt.savefig(os.path.join(fig_dir, fig_name + ".pdf"))
        plt.close()

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

    def report_video_stats(self, vehicle_counts: dict, 
                           vehicle_stop_counts: dict, 
                           vehicle_start_counts: dict) -> pd.DataFrame: 
    #cam_id, time, vehicle_type, count, stops, starts 
        # print(vehicle_counts, vehicle_stop_counts, vehicle_start_counts)
        count_df = pd.DataFrame.from_dict(vehicle_counts, 
                                          orient='index', columns = ['counts'],)
        stops_df = pd.DataFrame.from_dict(vehicle_stop_counts, 
                                          orient='index', columns = ['stops'])
        starts_df = pd.DataFrame.from_dict(vehicle_start_counts, 
                                          orient='index', columns= ['starts'])

        # combine into one dataframe
        stats_df = count_df.join([stops_df,starts_df], how = 'outer', sort=True).fillna(0)
        stats_df["video_name"] = self.video_name
        # rownames to a column
        stats_df.index.name = 'vehicle_type'
        stats_df.reset_index(inplace = True)
        # reorder columns
        stats_df = stats_df[['video_name', 'vehicle_type', 'counts', 'stops', 'starts']]

        return stats_df