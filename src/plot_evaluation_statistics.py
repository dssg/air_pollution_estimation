import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from traffic_analysis.d00_utils.get_project_directory import get_project_directory
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths
from traffic_viz.d06_visualisation.chunk_evaluation_plotting import (
    plot_video_stats_diff_distribution,
    plot_video_level_summary_stats,
    plot_mAP_over_time)


paths = load_paths()
creds = load_credentials()
params = load_parameters()
project_dir = get_project_directory()

data_location = os.path.join(project_dir, 'data', 'eval')
if not os.path.isdir(data_location):
    os.makedirs(data_location)

sqldata = DataLoaderSQL(paths=paths, creds=creds)
video_speed_cpu = sqldata.select_from_table('eval_video_performance2')
video_speed_gpu = sqldata.select_from_table('eval_video_performance_gpu')
video_diffs_cpu = sqldata.select_from_table('eval_video_stats2')
video_diffs_gpu = sqldata.select_from_table('eval_video_stats_gpu')
frame_map_cpu = sqldata.select_from_table('eval_frame_stats2')
frame_map_gpu = sqldata.select_from_table('eval_frame_stats_gpu')
video_speed_cpu.to_csv(path_or_buf=data_location + 'video_speed_cpu.csv', sep=',')
video_speed_gpu.to_csv(path_or_buf=data_location + 'video_speed_gpu.csv', sep=',')
video_diffs_cpu.to_csv(path_or_buf=data_location + 'video_diffs_cpu.csv', sep=',')
video_diffs_gpu.to_csv(path_or_buf=data_location + 'video_diffs_gpu.csv', sep=',')
frame_map_cpu.to_csv(path_or_buf=data_location + 'frame_map_cpu.csv', sep=',')
frame_map_gpu.to_csv(path_or_buf=data_location + 'frame_map_gpu.csv', sep=',')