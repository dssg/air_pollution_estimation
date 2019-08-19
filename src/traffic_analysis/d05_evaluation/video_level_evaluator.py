import pandas as pd
import numpy as np
import xml.etree.ElementTree as ElementTree

from traffic_analysis.d05_evaluation.parse_annotation import parse_annotation


class VideoLevelEvaluator:
    """
    Purpose of this class is to conduct video level evaluation for multiple videos.
    """
    def __init__(self,
                 videos_to_eval: pd.DataFrame,
                 video_level_df: pd.DataFrame,
                 video_level_column_order: list,
                 selected_labels: list):

        # data frames to work with
        self.videos_to_eval = videos_to_eval
        self.video_level_df = video_level_df
        self.video_level_ground_truth = pd.DataFrame({})

        # parameters
        self.video_level_column_order = video_level_column_order
        self.selected_labels = selected_labels
        self.stats_to_evaluate = ['counts', 'starts', 'stops']

    def evaluate(self) -> (pd.DataFrame, pd.DataFrame):
        """Get video level evaluation results 
        Returns: 
          performance_df -- data frame summarising the performance 
          diff_df -- data frame with raw differences of 
                     predictions vs ground truth
        """
        self.video_level_ground_truth = self.get_ground_truth()
        diff_df = self.compute_diffs()
        performance_df = self.summarise_performance(diff_df)
        return performance_df, diff_df

    def compute_diffs(self) -> pd.DataFrame:
        """Get raw diffs of pred minus true for each statistic we are 
        evaluating. 
        
        Returns:
            diff_df -- data frame with raw differences of 
                       predictions vs ground truth
        """
        # get data sets
        video_level_ground_truth = self.video_level_ground_truth
        video_level_estimates = pd.merge(left=self.videos_to_eval[['camera_id', 'video_upload_datetime']],
                                         right=self.video_level_df,
                                         on=['camera_id', 'video_upload_datetime'],
                                         how='inner')

        # calculate diffs
        truth_rename_dict = {stat: stat + '_true' for stat in self.stats_to_evaluate}
        pred_rename_dict = {stat: stat + '_pred' for stat in self.stats_to_evaluate}
        video_level_ground_truth.rename(columns=truth_rename_dict, inplace=True)
        video_level_estimates.rename(columns=pred_rename_dict, inplace=True)

        id_cols = ['camera_id', 'video_upload_datetime', 'vehicle_type']
        diff_df = pd.merge(left=video_level_ground_truth[id_cols + list(truth_rename_dict.values())],
                           right=video_level_estimates[id_cols + list(pred_rename_dict.values())],
                           on=id_cols,
                           how='left').fillna(0)
        for stat in self.stats_to_evaluate:
            diff_df[stat + '_diff'] = diff_df[stat + '_pred'] - diff_df[stat + '_true']
        return diff_df

    def summarise_performance(self, 
                              diff_df: pd.DataFrame) -> pd.DataFrame:
        # reshape for convenient summarising
        id_cols = ['camera_id', 'video_upload_datetime', 'vehicle_type']
        diff_cols = [stat + '_diff' for stat in self.stats_to_evaluate]
        diff_melted = diff_df.melt(id_vars=id_cols,
                                   value_vars=diff_cols,
                                   value_name='stat_diff',
                                   var_name='stat')
        diff_melted['stat'] = diff_melted['stat'].str.replace("_diff", "")

        # summarise
        performance_df = (diff_melted
                          .groupby(['vehicle_type', 'stat'])
                          .stat_diff
                          .agg({'bias': np.mean,
                                'MAE': lambda x: np.mean(abs(x)),
                                'RMSE': lambda x: np.sqrt(np.mean(x ** 2)),
                                'sd': np.std,
                                'n_videos': 'count'})
                          .reset_index())

        return performance_df

    def get_ground_truth(self):
        """Parse ground truth XMLs into corresponding ground truth 
        video level dataframe 
        """

        video_level_ground_truth_list = []
        for idx, video in self.videos_to_eval.iterrows():
            # get frame level ground truth
            xml_root = ElementTree.parse(video['xml_path']).getroot()
            frame_level_ground_truth = parse_annotation(xml_root)

            # get video level ground truth
            video_level_ground_truth = self.compute_video_level_ground_truth(frame_level_ground_truth)

            video_level_ground_truth['camera_id'] = video['camera_id']
            video_level_ground_truth['video_upload_datetime'] = video['video_upload_datetime']
            video_level_ground_truth_list.append(video_level_ground_truth)

        video_level_ground_truth = pd.concat(video_level_ground_truth_list, axis=0)

        missing_cols = list(set(self.video_level_column_order)
                            - set(video_level_ground_truth.columns))
        for column_name in missing_cols:
            video_level_ground_truth[column_name] = 0

        return video_level_ground_truth

    def compute_video_level_ground_truth(self,
                                         frame_level_ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Parse frame level ground truth dataframe into a video level ground 
        truth dataframe. Gets parked, starts, stops, counts
        """

        # prepare for evaluation
        ground_truth = frame_level_ground_truth.sort_values(['vehicle_id', 'frame_id'])
        ground_truth['parked'] = (ground_truth['parked']
                                  .replace({'false': False,
                                            'true': True}))
        ground_truth['stopped'] = (ground_truth['stopped']
                                   .replace({'false': False,
                                             'true': True}))

        ground_truth['motion_change'] = ((ground_truth['stopped'] -
                                          (ground_truth
                                           .groupby('vehicle_id')
                                           .stopped
                                           .shift(1)))
                                         .fillna(0)  # the first frame does not count as motion change
                                         )
        ground_truth['starts'] = (ground_truth['motion_change'] == -1).astype(int)
        ground_truth['stops'] = (ground_truth['motion_change'] == 1).astype(int)

        # aggregate to vehicle level
        vehicle_df = (ground_truth
                      .groupby('vehicle_id')
                      ['vehicle_type', 'parked']
                      .first()  # take initial status
                      .reset_index())

        start_stop = (ground_truth
                      .groupby(['vehicle_id'])
                      ['starts', 'stops']
                      .sum()
                      .reset_index())

        vehicle_df = pd.merge(left=vehicle_df,
                              right=start_stop,
                              on='vehicle_id',
                              how='inner')

        # aggregate to video level
        motion_df = (vehicle_df
                     .groupby('vehicle_type')
                     ['parked', 'starts', 'stops']
                     .sum()
                     .reset_index()
                     )
        motion_df['parked'] = motion_df['parked'].astype(int)

        count_df = (vehicle_df
                    .groupby('vehicle_type')
                    .vehicle_id
                    .count()
                    .reset_index()
                    .rename(columns={'vehicle_id': 'counts'})
                    )
        video_level_ground_truth = pd.merge(left=count_df,
                                            right=motion_df,
                                            on='vehicle_type',
                                            how='inner')

        # guarantee that all required types are reported
        all_types = pd.DataFrame({'vehicle_type': self.selected_labels})
        video_level_ground_truth = pd.merge(left=all_types,
                                            right=video_level_ground_truth,
                                            on='vehicle_type',
                                            how='left')
        for col in ['counts', 'parked', 'starts', 'stops']:
            video_level_ground_truth[col] = video_level_ground_truth[col].fillna(0).astype(int)

        return video_level_ground_truth
