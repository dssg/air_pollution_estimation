import numpy as np 
import pandas as pd 

from traffic_analysis.d05_evaluation.single_evaluator import SingleEvaluator
from traffic_analysis.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib
from traffic_analysis.d05_evaluation.compute_mean_average_precision import (
    get_avg_precision_at_iou,
    plot_pr_curve,
)


class FrameLevelEvaluator(SingleEvaluator):
    """
    Computes evaluation statistics from a frame_level_df corresponding to 
    one video. 

    Args: 
        xml_root: pointer to root of xml file
        xml_name: filename of annotation .xml
        frame_level_df: frame_level_df coresponding to one video, formatted as output by 
        the TrackingAnalyser
        params: parameter dictionary from config YAML
    """

    def __init__(self, 
                 xml_root, 
                 xml_name, 
                 frame_level_df: pd.DataFrame, 
                 params: dict):
        super().__init__(xml_root, xml_name, params)
        self.ground_truth_df = super().parse_annotation()
        self.frame_level_df = frame_level_df
        self.vehicle_types = params["selected_labels"]
        self.n_frames = len(frame_level_df["frame_id"].unique())

    def evaluate_video(self) -> pd.DataFrame:
        """Performs frame level evaluation on one video; i.e. computes mean average 
        precision for the video 

        Args: 
        Returns: 
            mAP_df: contains mean average precision for each vehicle type in the video  
        Raises: 
        """

        self.ground_truth_df = self.ground_truth_df.sort_values(
            by="frame_id"
        ).reset_index()
        self.frame_level_df = self.frame_level_df.sort_values(
            by="frame_id"
        ).reset_index()

        # need to delete bboxes of [0,0,0,0] from frame_level_df
        zeros_mask = self.frame_level_df.bboxes.apply(
            lambda x: all(True if bbox_entry == 0.0 else False for bbox_entry in x)
        )
        self.frame_level_df = self.frame_level_df[~zeros_mask].reset_index()

        ground_truth_dict = self.reparse_bboxes_df(
            self.ground_truth_df, include_confidence=False
        )
        predicted_dict = self.reparse_bboxes_df(
            self.frame_level_df, include_confidence=True, bbox_format="cv2"
        )

        mAP_dict = self.compute_mAP_video(ground_truth_dict, predicted_dict)
        mAP_df = pd.DataFrame.from_dict(
            mAP_dict, orient="index", columns=["mean_avg_precision"]
        )

        assert (self.frame_level_df["camera_id"].iloc[0] == self.camera_id),\
              "camera IDs from annotation file and frame_level_df do not match \
               in FrameLevelEvaluator.evaluate_video()"
        mAP_df["camera_id"] = self.camera_id

        assert (self.frame_level_df["video_upload_datetime"].iloc[0]
            == self.video_upload_datetime), \
            "dates from annotation file and frame_level_df do not match \
            in FrameLevelEvaluator.evaluate_video()"
        mAP_df["video_upload_datetime"] = self.video_upload_datetime

        mAP_df.index.name = "vehicle_type"
        mAP_df.reset_index(inplace=True)

        return mAP_df

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
