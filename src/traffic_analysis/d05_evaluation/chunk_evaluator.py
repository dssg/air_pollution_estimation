# testing imports 
import os 
from traffic_analysis.d00_utils.load_confs import load_parameters
###################
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d05_evaluation.video_level_evaluator import VideoLevelEvaluator
from traffic_analysis.d05_evaluation.frame_level_evaluator import FrameLevelEvaluator

import pandas as pd
import re


class ChunkEvaluator:
    """
    Purpose of this class is to evaluate a chunk of videos, given
    a list of the annotation_xml_paths, and lists of the corresponding
    video_level_dfs and frame_level_dfs.
    """
    def __init__(self,
                 annotation_xml_paths: list,
                 params: dict,
                 video_level_df: pd.DataFrame=None,
                 frame_level_df: pd.DataFrame=None):

        annotations_available = {}
        for xml_path in annotation_xml_paths:
            xml_name = re.split(r"\\|/", xml_path)[-1]
            camera_id, video_upload_datetime = parse_video_or_annotation_name(xml_name)
            annotations_available[xml_path] = [camera_id, video_upload_datetime]

        annotations_available = (pd.DataFrame
                                 .from_dict(annotations_available,
                                            orient='index',
                                            columns=['camera_id', 'video_upload_datetime'])
                                 .reset_index()
                                 .rename(columns={'index': 'xml_path'}))

        if video_level_df is not None:
            video_level_videos_to_eval = video_level_df[['camera_id', 'video_upload_datetime']].drop_duplicates()

            # evaluate only those videos for which we have annotations
            self.video_level_videos_to_eval = pd.merge(left=annotations_available,
                                           right=video_level_videos_to_eval,
                                           on=['camera_id', 'video_upload_datetime'],
                                           how='inner')

            self.num_video_level_videos = len(self.video_level_videos_to_eval)

            assert self.num_video_level_videos > 0
            self.video_level_df = video_level_df

            self.video_level_column_order = params['video_level_column_order']
            self.selected_labels = params['selected_labels']

        if frame_level_df is not None: 
            frame_level_videos_to_eval = frame_level_df[['camera_id', 'video_upload_datetime']].drop_duplicates()

            # evaluate only those frame level videos for which we have annotations
            self.frame_level_videos_to_eval = pd.merge(left=annotations_available,
                                           right=frame_level_videos_to_eval,
                                           on=['camera_id', 'video_upload_datetime'],
                                           how='inner')
            self.num_frame_level_videos = len(self.video_level_videos_to_eval)

            assert self.num_frame_level_videos > 0
            self.frame_level_df = frame_level_df
            # self.video_level_column_order = params['video_level_column_order']

        self.params = params

    def evaluate_video_level(self)-> pd.DataFrame:
        """This function evaluates a chunk of videos utilizing multiple SingleEvaluator
           objects.
        """


        video_level_evaluator = VideoLevelEvaluator(videos_to_eval=self.video_level_videos_to_eval,
                                                    video_level_df=self.video_level_df,
                                                    video_level_column_order=self.video_level_column_order,
                                                    selected_labels=self.selected_labels)

        video_level_performance, video_level_diff_df = video_level_evaluator.evaluate()
        return video_level_performance, video_level_diff_df

    # # frame level evaluation
    # def evaluate_frame_level(self):
    #     frame_level_mAP_dfs = []
    #     for i, xml_path in enumerate(self.annotation_xml_paths):
    #         xml_name = re.split(r"\\|/", xml_path)[-1]
    #         xml_root = ElementTree.parse(xml_path).getroot()
    #         frame_level_evaluator = FrameLevelEvaluator(xml_root, xml_name,
    #                                                     self.frame_level_dfs[i],
    #                                                     self.params)
    #         frame_level_mAP_dfs.append(frame_level_evaluator.evaluate_video())
    #     frame_level_mAP_df = pd.concat(
    #         frame_level_mAP_dfs, axis=0)  # concat dfs as new rows
    #     return frame_level_mAP_df

if __name__ == '__main__':
    params = load_parameters()
    pd.set_option('display.max_columns', 500)

    annotations_dir = "C:\\Users\\Caroline Wang\\OneDrive\\DSSG\\air_pollution_estimation\\annotations"
    xml_paths = [os.path.join(annotations_dir, '14_2019-06-29_13-01-19.744908_00001.05900.xml'),
                 os.path.join(annotations_dir, '15_2019-06-29_13-01-03.094068_00001.01252.xml')]

    video_level_df = pd.read_csv("../data/carolinetemp/video_level_df.csv",
                        dtype={"camera_id": str},
                        parse_dates=["video_upload_datetime"])
    del video_level_df['Unnamed: 0']


    chunk_evaluator = ChunkEvaluator(annotation_xml_paths=xml_paths,
                                     params=params,
                                     video_level_df=video_level_df)
    video_level_performance, video_level_diff_df = chunk_evaluator.evaluate_video_level()

    print(video_level_performance)
    print(video_level_diff_df)