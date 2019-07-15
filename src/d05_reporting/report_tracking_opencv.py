from src.d04_modelling.vehiclefleet import VehicleFleet
import numpy as np 
import pandas as pd 
import os 


def construct_frame_level_table_tracking(fleet:VehicleFleet) -> pd.DataFrame: 
    """Wrapper function for creating a table with frame level stats. See VehicleFleet 
    class for more information. 
    """
    return fleet.report_frame_level_info()


def reconstruct_fleet_from_frame_level_table(frame_level_df:pd.DataFrame) -> VehicleFleet:
    """Wrapper function for reconstructing a fleet object from a frame_level_df. See VehicleFleet 
    class for more information.  
    """
    return VehicleFleet(frame_level_df = frame_level_df, load_from_pd = True)


def construct_video_level_table_tracking(fleet:VehicleFleet,
                                         smoothing_method:str,
                                         iou_convolution_window:int,
                                         stop_start_iou_threshold:float) -> pd.DataFrame:
    """Construct video-level stats table using tracking techniques 

    Keyword arguments: 
    fleet -- instance of the VehicleFleet class, which stores tracking/obj detection info 
             of all the vehicles in a video
    iou_convolution_window -- convolution window size when computing the iou between two bboxes
    smoothing_method -- method to use to smooth the iou time series; see stats_helpers.py
    stop_start_iou_threshold -- iou threshold between 0 and 1; above this, we consider the vehicle 
                                as stopped; below, we consider the vehicle moving
    """
    # compute the convolved IOU time series for each vehicle and smooth
    fleet.compute_iou_time_series(interval=iou_convolution_window)
    fleet.smooth_iou_time_series(smoothing_method=smoothing_method)
    # sample plotting options 
    # fleet.plot_iou_time_series(fig_dir="data", fig_name="param_tuning", smoothed=True)
    video_level_df = fleet.report_video_level_stats(fleet.compute_counts(), 
                                       *fleet.compute_stop_starts(stop_start_iou_threshold))
    return video_level_df
