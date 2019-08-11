import pandas as pd 
import numpy as np
import xml.etree.ElementTree as ElementTree

from traffic_analysis.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib
from traffic_analysis.d05_evaluation.parse_annotation import parse_annotation
from traffic_analysis.d05_evaluation.compute_mean_average_precision import get_avg_precision_at_iou


class FrameLevelEvaluator:
    """
    Conduct frame level evaluation for one video.
    """
    def __init__(self,
                 videos_to_eval: pd.DataFrame,
                 frame_level_df: pd.DataFrame,
                 selected_labels: list,
                 data_loader_s3: None):

        # data frames to work with
        self.videos_to_eval = videos_to_eval
        self.frame_level_df = frame_level_df
        self.frame_level_ground_truth = pd.DataFrame({})
        self.frame_level_preds = pd.DataFrame({})

        # parameters
        self.selected_labels = selected_labels
        if data_loader_s3 is not None: 
            self.from_s3_paths = True
            self.dl_s3 = data_loader_s3
        else:
            self.from_local_paths = True

    def evaluate(self) -> pd.DataFrame:
        """Compute mean average precision for each vehicle type on multiple videos 
        """
        self.frame_level_ground_truth = self.get_ground_truth()
        self.frame_level_preds = self.filter_frame_level_df()

        frame_level_map_dfs = []
        for (gt_camera_id, gt_video_upload_datetime), ground_truth_df in \
            self.frame_level_ground_truth.groupby(["camera_id", "video_upload_datetime"]):
            # get corresponding predictions for this video 
            pred_df = self.frame_level_preds[(self.frame_level_preds["camera_id"] == gt_camera_id) &
                                             (self.frame_level_preds["video_upload_datetime"] ==
                                              gt_video_upload_datetime)].copy()
            pred_frame_max = int(pred_df["frame_id"].max()) 
            pred_frame_min = int(pred_df["frame_id"].min())

            ground_truth_frame_max = int(ground_truth_df["frame_id"].max()) 
            ground_truth_frame_min = int(ground_truth_df["frame_id"].min())

            print("camera id and video_upload_datetime is ", gt_camera_id, gt_video_upload_datetime)
            print(f"pred frame min is {pred_frame_min}, gt frame min is {ground_truth_frame_min}\n \
                    pred frame max is {pred_frame_max}, gt frame max is {ground_truth_frame_max}")

            num_gt_frames = ground_truth_df["frame_id"].nunique()
            num_pred_frames = pred_df["frame_id"].nunique()

            try: 
                assert num_gt_frames == num_pred_frames, \
                    f"Number of unique frames in predicted df is {num_pred_frames}, \
                    number of unique frames in ground truth df is {num_gt_frames}."

            except: 
                print("Assertion failed: camera id and video_upload_datetime is ", gt_camera_id, gt_video_upload_datetime)

            max_frame_ind = ground_truth_df["stop_frame"].iloc[0]
            ground_truth_dict = self.reparse_bboxes_df(ground_truth_df, 
                                                       max_frame_ind = max_frame_ind,
                                                       include_confidence=False)
            predicted_dict = self.reparse_bboxes_df(pred_df,
                                                    max_frame_ind = max_frame_ind,
                                                    include_confidence=True, 
                                                    bbox_format="cv2")

            map_dict = self.compute_map_video(ground_truth_dict, predicted_dict)
            map_df = pd.DataFrame.from_dict(map_dict, 
                                            orient="index", 
                                            columns=["mean_avg_precision"])

            map_df["camera_id"] = gt_camera_id
            map_df["video_upload_datetime"] = gt_video_upload_datetime

            frame_level_map_dfs.append(map_df)

        frame_level_map_df = pd.concat(frame_level_map_dfs, 
                                       axis=0)  
        frame_level_map_df.index.name = "vehicle_type"
        frame_level_map_df.reset_index(inplace=True)

        return frame_level_map_df

    def filter_frame_level_df(self) -> pd.DataFrame:
        """
        Get preds for videos which are in videos_to_eval
        """
        frame_level_df_filt = pd.merge(left=self.videos_to_eval[['camera_id', 'video_upload_datetime']],
                                       right=self.frame_level_df,
                                       on=['camera_id', 'video_upload_datetime'],
                                       how='inner')

        zeros_mask = frame_level_df_filt.bboxes.apply(
            lambda x: all(True if bbox_entry == 0.0 else False for bbox_entry in x))

        frame_level_df_filt = (frame_level_df_filt[~zeros_mask]
                               .sort_values(by=["camera_id", "video_upload_datetime"])
                               .reset_index(drop=True))
        return frame_level_df_filt

    def get_ground_truth(self) -> pd.DataFrame:
        """Read in annotation xmls from paths stored in self.videos_to_eval
        Frames are 0-indexed. 
        """
        frame_level_ground_truth_dfs = []
        for idx, video in self.videos_to_eval.iterrows():
            # get frame level ground truth
            if self.from_s3_paths:
                xml_root = self.dl_s3.read_xml(video['xml_path'])
            elif self.from_local_paths: # read from local  
                xml_root = ElementTree.parse(video['xml_path']).getroot()

            frame_level_ground_truth = parse_annotation(xml_root)
            frame_level_ground_truth["camera_id"] = video["camera_id"]
            frame_level_ground_truth["video_upload_datetime"] = video["video_upload_datetime"]
            frame_level_ground_truth_dfs.append(frame_level_ground_truth)

        frame_level_ground_truth = pd.concat(frame_level_ground_truth_dfs,
                                             axis=0)
        return frame_level_ground_truth

    def reparse_bboxes_df(self, 
                          df: pd.DataFrame, 
                          max_frame_ind: int,
                          include_confidence: bool, 
                          bbox_format: str = "cvlib") -> dict:
        """Restructures dfs containing bboxes for each frame (i.e. frame level df, 
        ground truth df) to a dictionary of dictionaries. This format is what 
        compute_mean_average_precision.py functions take as input. 

        This function also ensures that every frame in the video has a corresponding 
        dict entry (even if the input df had no prediction for that frame)

        Args: 
            df: frame_level_df which contains bboxes corresponding to each frame of
                a video. 
            max_frame_ind: index of the last frame. We assume that frames are 0 indexed, so the 
                           total num frames in the video should be max_frame_ind + 1
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
            bboxes_cvlib = pd.Series(bboxcv2_to_bboxcvlib(bboxes_np, vectorized=True).tolist()).values
            df.loc[:, "bboxes"] = bboxes_cvlib

        # initialize dictionaries to correct shape
        if include_confidence:
            df_as_dict = {
                vehicle_type: {
                    "frame" + str(i): {"bboxes": [], "scores": []}
                    for i in range(max_frame_ind + 1)
                }
                for vehicle_type in self.selected_labels
            }

        else:
            df_as_dict = {
                vehicle_type: {"frame" + str(i): [] for i in range(max_frame_ind + 1)}
                for vehicle_type in self.selected_labels
            }

        for (vehicle_type, frame_id), vehicle_frame_df in df.groupby(
                ["vehicle_type", "frame_id"]):

            frame_id = int(frame_id)

            if vehicle_type not in self.selected_labels: 
                continue
            if frame_id > max_frame_ind:
                print("Warning: more frames in vehice_frame_df than max_frame_id")

            # fill dictionary
            if include_confidence:
                df_as_dict[vehicle_type]["frame" + str(frame_id)]["bboxes"] = \
                    vehicle_frame_df["bboxes"].tolist()
                df_as_dict[vehicle_type]["frame" + str(frame_id)]["scores"] = \
                    vehicle_frame_df["confidence"].tolist()
            else:
                df_as_dict[vehicle_type]["frame" + str(frame_id)] = \
                    vehicle_frame_df["bboxes"].tolist()

        return df_as_dict

    def compute_map_video(self, ground_truth_dict, predicted_dict) -> dict:
        """ Function computes the mean average precision for each vehicle type for a video
        Args: 
            ground_truth_dict: ground_truth_df reparsed by reparse_bboxes_df
            predicted_dict: frame_level_df reparsed by reparse_bboxes_df
        Returns: 
            map_dict: dictionary with vehicle_types as keys and maps as values 
        Raises: 
        """
        map_dict = {vehicle_type: -1.0 for vehicle_type in self.selected_labels}

        for vehicle_type in self.selected_labels:
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

            # avg the avg precision for each IOU value
            mean_avg_precision = 100 * np.mean(avg_precs)
            map_dict[vehicle_type] = mean_avg_precision

        return map_dict
