import pandas as pd 
import numpy as np

import xml.etree.ElementTree as ElementTree
from traffic_analysis.d05_evaluation.parse_annotation import parse_annotation
from traffic_analysis.d05_evaluation.compute_mean_average_precision import (
    get_avg_precision_at_iou,
    plot_pr_curve,
)


class FrameLevelEvaluator:
    """
    Purpose of this class is to conduct video level evaluation for one video.
    """
    def __init__(self,
                 videos_to_eval: pd.DataFrame,
                 frame_level_df: pd.DataFrame,
                 frame_level_column_order: list,
                 selected_labels: list):

        # data frames to work with
        self.videos_to_eval = videos_to_eval
        self.frame_level_df = frame_level_df
        self.frame_level_ground_truth = pd.DataFrame({})

        # parameters
        self.frame_level_column_order = frame_level_column_order
        self.selected_labels = selected_labels


    def evaluate(self):
        pass

    def get_ground_truth(self) -> dict:
        frame_level_ground_truth_list = []
        for idx, video in self.videos_to_eval.iterrows():
            # get frame level ground truth
            xml_root = ElementTree.parse(video['xml_path']).getroot()
            frame_level_ground_truth = parse_annotation(xml_root)

            # get frame level ground truth
            frame_level_ground_truth = self.compute_frame_level_ground_truth(frame_level_ground_truth)
                                           .sort_values(by="frame_id")
                                           .reset_index()

            frame_level_ground_truth['camera_id'] = video['camera_id']
            frame_level_ground_truth['video_upload_datetime'] = video['video_upload_datetime']
            frame_level_ground_truth_list.append(frame_level_ground_truth)

        frame_level_ground_truth = pd.concat(frame_level_ground_truth_list, axis=0)

        missing_cols = list(set(self.frame_level_column_order)
                            - set(frame_level_ground_truth.columns))
        for column_name in missing_cols:
            frame_level_ground_truth[column_name] = 0

        return frame_level_ground_truth

    def compute_frame_level_ground_truth(self):
        pass

    def reparse_bboxes_df(self, 
                          df: pd.DataFrame, 
                          include_confidence: bool, 
                          bbox_format: str = "cvlib") -> dict:
        """Restructures dfs containing bboxes for each frame (i.e. frame level df, 
        ground truth df) to a dictionary of dictionaries. This format is what 
        compute_mean_average_precision.py functions take as input. 

        Args: 
            df: frame_level_df which contains bboxes corresponding to each frame of
                a video. 
            include_confidence: If this df contains the confidence corresponding to 
                                the bbox predictions, this should be specified (the 
                                reparser will construct a sub-dict for this case)
            bbox_format: cvlib is cvlib (xmin,ymin, xmin+width, ymin+height), 
                         cv2 is (xmin,ymin,width,height)
        Returns: df as a nested dictionary  
        Raises: 
        """
        # dict of dict of dicts, with outermost layer being the vehicle type
        bboxes_np = np.array(df["bboxes"].values.tolist())
        assert bboxes_np.shape[1] == 4

        if bbox_format == "cv2":
            # convert to format cvlib
            df["bboxes"] = pd.Series(
                bboxcv2_to_bboxcvlib(bboxes_np, vectorized=True).tolist()
            )

        # initialize dictionaries to correct shape
        if include_confidence:
            df_as_dict = {
                vehicle_type: {
                    "frame" + str(i): {"bboxes": [], "scores": []}
                    for i in range(self.n_frames)
                }
                for vehicle_type in self.vehicle_types
            }

        else:
            df_as_dict = {
                vehicle_type: {"frame" + str(i): [] for i in range(self.n_frames)}
                for vehicle_type in self.vehicle_types
            }

        # fill dictionary
        for (vehicle_type, frame_id), vehicle_frame_df in df.groupby(
            ["vehicle_type", "frame_id"]
        ):
            if include_confidence:
                df_as_dict[vehicle_type]["frame" + str(frame_id)]["bboxes"] = \
                    vehicle_frame_df["bboxes"].tolist()
                df_as_dict[vehicle_type]["frame" + str(frame_id)]["scores"] = \
                    vehicle_frame_df["confidence"].tolist()
            else:
                df_as_dict[vehicle_type]["frame" + str(frame_id)] = \
                vehicle_frame_df["bboxes"].tolist()

        return df_as_dict

    def compute_mAP_video(self, ground_truth_dict, predicted_dict) -> dict:
        """ Function computes the mean average precision for each vehicle type for a video
        Args: 
            ground_truth_dict: ground_truth_df reparsed by reparse_bboxes_df
            predicted_dict: frame_level_df reparsed by reparse_bboxes_df
        Returns: 
            mAP_dict: dictionary with vehicle_types as keys and mAPs as values 
        Raises: 
        """
        mAP_dict = {vehicle_type: -1.0 for vehicle_type in self.vehicle_types}

        for vehicle_type in self.vehicle_types:
            vehicle_gt_dict = ground_truth_dict[vehicle_type]
            vehicle_pred_dict = predicted_dict[vehicle_type]

            avg_precs = []
            iou_thrs = []
            # compute avg precision for 10 IOU thresholds from .5 to .95 (COCO challenge standard)
            for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
                data_dict = get_avg_precision_at_iou(
                    gt_bboxes=vehicle_gt_dict,
                    pred_bboxes=vehicle_pred_dict,
                    iou_thr=iou_thr,
                )
                avg_precs.append(data_dict["avg_prec"])
                iou_thrs.append(iou_thr)

                precisions = data_dict["precisions"]
                recalls = data_dict["recalls"]

            avg_precs = [float("{:.4f}".format(ap)) for ap in avg_precs]
            iou_thrs = [float("{:.4f}".format(thr)) for thr in iou_thrs]

            # avg the avg precision for each IOU value
            mean_avg_precision = 100 * np.mean(avg_precs)
            mAP_dict[vehicle_type] = mean_avg_precision

        return mAP_dict
