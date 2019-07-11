# from src.d04_modelling.vehiclefleet import VehicleFleet
import numpy as np 
import pandas as pd 
import os 
import pickle as pkl


def construct_frame_level_table_tracking(fleet:VehicleFleet) -> pd.DataFrame: 
	return 


def construct_video_level_table_tracking(fleet:VehicleFleet) -> pd.DataFrame
	"""
	"""
    # compute the convolved IOU time series for each vehicle and smooth
    fleet.compute_iou_time_series(interval=iou_convolution_window)
    fleet.smooth_iou_time_series(smoothing_method=smoothing_method)
    # fleet.plot_iou_time_series(fig_dir="data", fig_name="param_tuning", smoothed=True)
    stats_df = fleet.report_video_stats(fleet.compute_counts(), 
    									*fleet.compute_stop_starts(stop_start_iou_threshold))
    return stats_df


if __name__ == '__main__':
	with open('data/pickled/fleet_obj.pkl', 'rb') as handle: 
        fleet = pkl.load(handle)

    print(fleet.video_name)
    