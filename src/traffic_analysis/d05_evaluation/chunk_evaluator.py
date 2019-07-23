from traffic_analysis.d05_evaluation.single_evaluator import FrameLevelEvaluator, VideoLevelEvaluator
from traffic_analysis.d00_utils.load_confs import load_parameters

import numpy as np 
import pandas as pd
import glob
import re 
import xml.etree.ElementTree as ElementTree
from functools import reduce 

class ChunkEvaluator(): 
    """
    Purpose of this class is to evaluate a chunk of videos, given 
    a list of the annotation_xml_paths, and lists of the corresponding 
    video_level_dfs and frame_level_dfs. 
    """
    def __init__(self,
                 annotation_xml_paths: list,
                 params: dict,
                 frame_level_dfs: list = None,
                 video_level_dfs: list = None):
        self.num_videos = len(annotation_xml_paths)
        self.annotation_xml_paths = annotation_xml_paths

        if frame_level_dfs is not None: 
            assert self.num_videos == len(frame_level_dfs) 
            self.frame_level_dfs = frame_level_dfs

        if video_level_dfs is not None: 
            assert self.num_videos == len(video_level_dfs)
            self.video_level_dfs = video_level_dfs

        self.params = params

    def evaluate_videos(self)->pd.DataFrame: 
        """This function evaluates a chunk of videos utilizing multiple SingleEvaluator objects. 
        """
        video_level_diff_dfs = []
        for i, xml_path in enumerate(self.annotation_xml_paths):
            xml_name = re.split(r"\\|/", xml_path)[-1]
            xml_root = ElementTree.parse(xml_path).getroot()
            video_level_evaluator = VideoLevelEvaluator(xml_root, xml_name,
                                                        self.video_level_dfs[i],
                                                        params)
            video_level_diff_dfs.append(video_level_evaluator.evaluate_video())
        video_level_diff_df = pd.concat(video_level_diff_dfs, axis=0)  # concat dfs as new rows

        all_vehicles_dfs = []

        for vehicle_type, vehicle_type_df in video_level_diff_df.groupby(["vehicle_type"]):
            single_vehicle_df = self.video_statistics(vehicle_type_df, self.params['video_level_stats'])
            single_vehicle_df['vehicle_type'] = vehicle_type

            all_vehicles_dfs.append(single_vehicle_df)

        # put into single df
        all_vehicles_df = pd.concat(all_vehicles_dfs, axis=0)
        all_vehicles_df['n_videos'] = self.num_videos
        all_vehicles_df.reset_index(inplace=True)
        del all_vehicles_df['index']
        return all_vehicles_df

    def video_statistics(self, df:pd.DataFrame, vehicle_stat_cols:list): 
        """Computes mse, standard deviation, and mean difference across all vehicle types 
        for a chunk of videos
        """
        vehicle_stats_dfs = []
        for vehicle_stat in vehicle_stat_cols: # counts, starts, stops, parked 
            vehicle_stat_dict = {'statistic': ['mean_diff', 'mse', 'sd']}

            #compute mean diff
            mean_diff = np.sum(df["y_pred-y_"+vehicle_stat].values)/self.num_videos
            #compute MSE
            mse = np.sum(np.square(df["y_pred-y_"+vehicle_stat].values))/self.num_videos
            #compute SD
            sd = np.std(df["y_pred-y_"+vehicle_stat].values)

            vehicle_stat_dict[vehicle_stat] = [mean_diff, mse, sd]
            vehicle_stat_df = pd.DataFrame.from_dict(vehicle_stat_dict)

            vehicle_stats_dfs.append(vehicle_stat_df)
        vehicle_stats_df = reduce(lambda left,right: pd.merge(left, right, on=['statistic'],
                                  how='outer'), vehicle_stats_dfs)

        return vehicle_stats_df

    def evaluate_frame_level(self): pass 

    def frame_statistics(self): pass

    def plot_evaluation_stats(self): pass

if __name__ == '__main__':
    params = load_parameters()
    xml_paths = ["C:\\Users\\Caroline Wang\\OneDrive\\DSSG\\air_pollution_estimation\\annotations\\15_2019-06-29_13-01-03.094068_00001.01252.xml",
                "C:\\Users\\Caroline Wang\\OneDrive\\DSSG\\air_pollution_estimation\\annotations\\14_2019-06-29_13-01-19.744908_00001.05900.xml"]

    pd.set_option('display.max_columns', 500)

    video_level_dfs = pd.read_csv("../data/carolinetemp/video_level_df.csv",
                        dtype={"camera_id": str},
                        parse_dates=["video_upload_datetime"])
    del video_level_dfs['Unnamed: 0']

    video_level_df_list = []
    video_level_groups = video_level_dfs.groupby(['camera_id', 'video_upload_datetime'])
    for name, group in video_level_groups: 
        video_level_df_list.append(group)

    chunk_evaluator = ChunkEvaluator(annotation_xml_paths=xml_paths,
                                     params=params,
                                     video_level_dfs=video_level_df_list)
    stats_df = chunk_evaluator.evaluate_videos()
    print("stats df: \n", stats_df)