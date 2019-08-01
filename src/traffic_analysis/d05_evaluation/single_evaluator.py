import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d00_utils.bbox_helpers import bboxcv2_to_bboxcvlib
from traffic_analysis.d05_evaluation.compute_mean_average_precision import (
    get_avg_precision_at_iou,
    plot_pr_curve,
)


class SingleEvaluator:
    """
    This is a superclass for FrameLevelEvaluator and VideoLevelEvaluator
    """

    def __init__(self, xml_root, xml_name, params: dict):
        self.xml_root = xml_root
        self.camera_id, self.video_upload_datetime = parse_video_or_annotation_name(
            xml_name
        )
        self.annotated_result = {
            "vehicle_id": [],
            "frame_id": [],
            "bboxes": [],
            "vehicle_type": [],
            "parked": [],
            "stopped": [],
        }
        self.selected_labels = params["selected_labels"]

    def parse_annotation(self) -> pd.DataFrame:
        """Returns annotation xml file as a pandas dataframe
        """
        for track in self.xml_root.iter("track"):
            if track.attrib["label"] == "vehicle":
                for frame in track.iter("box"):
                    self.annotated_result["vehicle_id"].append(int(track.attrib["id"]))
                    self.annotated_result["frame_id"].append(int(frame.attrib["frame"]))
                    self.annotated_result["bboxes"].append(
                        [
                            float(frame.attrib["xtl"]),
                            float(frame.attrib["ytl"]),
                            float(frame.attrib["xbr"]),
                            float(frame.attrib["ybr"]),
                        ]
                    )
                    for attr in frame.iter("attribute"):
                        # If name is 'type' then index the dictionary using 'vehicle_type'
                        if attr.attrib["name"] == "type":
                            self.annotated_result["vehicle_type"].append(attr.text)
                        # Else just use the name for indexing
                        else:
                            self.annotated_result[attr.attrib["name"]].append(attr.text)

        ground_truth_df = pd.DataFrame.from_dict(self.annotated_result)
        ground_truth_df["video_upload_datetime"] = self.video_upload_datetime
        ground_truth_df["camera_id"] = self.camera_id
        return ground_truth_df


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

    def __init__(self, xml_root, xml_name, frame_level_df: pd.DataFrame, params: dict):
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

        assert (
            self.frame_level_df["camera_id"].iloc[0] == self.camera_id
        ), "camera IDs from annotation file and frame_level_df do not match \
               in FrameLevelEvaluator.evaluate_video()"
        mAP_df["camera_id"] = self.camera_id

        assert (
            self.frame_level_df["video_upload_datetime"].iloc[0]
            == self.video_upload_datetime
        ), "dates from annotation file and frame_level_df do not match \
            in FrameLevelEvaluator.evaluate_video()"
        mAP_df["video_upload_datetime"] = self.video_upload_datetime

        mAP_df.index.name = "vehicle_type"
        mAP_df.reset_index(inplace=True)

        return mAP_df

    def reparse_bboxes_df(
        self, df: pd.DataFrame, include_confidence: bool, bbox_format: str = "cvlib"
    ) -> dict:
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
                df_as_dict[vehicle_type]["frame" + str(frame_id)][
                    "bboxes"
                ] = vehicle_frame_df["bboxes"].tolist()
                df_as_dict[vehicle_type]["frame" + str(frame_id)][
                    "scores"
                ] = vehicle_frame_df["confidence"].tolist()
            else:
                df_as_dict[vehicle_type]["frame" + str(frame_id)] = vehicle_frame_df[
                    "bboxes"
                ].tolist()

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


####################################
class VideoLevelEvaluator(SingleEvaluator):
    """
    Purpose of this class is to conduct video level evaluation for one video.
    """

    def __init__(self, xml_root, xml_name, video_level_df: pd.DataFrame, params: dict):
        super().__init__(xml_root, xml_name, params)
        self.ground_truth_df = super().parse_annotation()
        self.video_level_df = video_level_df
        self.video_level_column_order = params["video_level_column_order"]

    def evaluate_video(self) -> pd.DataFrame:
        """Conducts evaluation on one video. Compares a true_stats_df generated from the 
        xml files with a video_level_df output by an instance of TrafficAnalyser. 
        """
        true_stats_df = self.compute_true_video_level_stats()
        true_stats_df = self.fill_and_sort_by_vehicle_types(true_stats_df)
        self.video_level_df = self.fill_and_sort_by_vehicle_types(self.video_level_df)

        diff_columns = [
            col if i < 3 else "y_pred-y_" + col
            for i, col in enumerate(self.video_level_column_order)
        ]
        diff_df = pd.DataFrame(columns=diff_columns)

        for stat in self.video_level_column_order[3:]:
            diff_df["y_pred-y_" + stat] = (
                self.video_level_df[stat] - true_stats_df[stat]
            )

        assert (
            self.video_level_df["camera_id"].iloc[0]
            == true_stats_df["camera_id"].iloc[0]
        ), "camera IDs do not match in VideoLevelEvaluator.evaluate_video()"
        diff_df["camera_id"] = self.video_level_df["camera_id"]

        assert (
            self.video_level_df["video_upload_datetime"].iloc[0]
            == true_stats_df["video_upload_datetime"].iloc[0]
        ), "dates do not match in VideoLevelEvaluator.evaluate_video()"
        diff_df["video_upload_datetime"] = self.video_level_df["video_upload_datetime"]

        diff_df["vehicle_type"] = self.video_level_df["vehicle_type"]
        return diff_df

    def fill_and_sort_by_vehicle_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures that the df passed in has all the vehicle types we care about, 
        and all the stats we care about (ex. counts, starts, stops, parked). If any of
        these are missing, this function fills the missing values with 0. Also sorts the 
        df by vehicle_types. 
        """
        for column_name in self.video_level_column_order:
            if column_name not in df.columns:
                df[column_name] = 0
        df = df[self.video_level_column_order]

        # insert missing vehicle types as row
        current_vehicle_types = df["vehicle_type"].values
        for vehicle_type in self.selected_labels:
            # need to append rows
            if vehicle_type not in current_vehicle_types:
                # append type as new column
                new_row_dict = {
                    column_name: (df[column_name].iloc[0] if i < 3 else 0)
                    for i, column_name in enumerate(self.video_level_column_order)
                }
                new_row_dict["vehicle_type"] = vehicle_type
                df.loc[len(df) + 1] = new_row_dict
        df = df.sort_values(by="vehicle_type").reset_index()
        return df

    def compute_true_video_level_stats(self) -> pd.DataFrame:
        """
        Combines counts, parked, stop-starts into one dataframe. 
        """
        counts_df = self.get_true_vehicle_counts()
        parked_df = self.get_true_parked_counts()
        stops_starts_df = self.get_true_stop_start_counts()

        counts_df["camera_id"] = self.camera_id
        counts_df["video_upload_datetime"] = self.video_upload_datetime
        counts_df = counts_df.merge(parked_df, how="outer", on="vehicle_type")
        counts_df = counts_df.merge(
            stops_starts_df, how="outer", on="vehicle_type"
        ).fillna(0)

        counts_df = counts_df[self.video_level_column_order]
        return counts_df

    def get_true_vehicle_counts(self) -> pd.DataFrame:
        """Report the true counts for one annotated videos.
        """
        types = self.ground_truth_df.groupby("vehicle_id")["vehicle_type"].unique()
        types = [t[0] for t in types]

        vals, counts = np.unique(types, return_counts=True)
        counts_df = pd.DataFrame(counts, index=vals)

        counts_df.index.name = "vehicle_type"
        counts_df.reset_index(inplace=True)

        counts_df = counts_df.rename({0: "counts"}, axis="columns").fillna(0)

        return counts_df

    def get_true_stop_start_counts(self) -> (pd.DataFrame, pd.DataFrame):
        """ Computes the number of stops for each vehicle from the ground_truth_df
        """
        stop_counts, start_counts = [], []
        vehicle_dfs = self.ground_truth_df.sort_values(
            ["frame_id"], ascending=True
        ).groupby("vehicle_id")

        # compute number of starts stops for each vehicle in the video
        for vehicle_id, vehicle_df in vehicle_dfs:
            vehicle_type = vehicle_df["vehicle_type"].tolist()[0]
            bool_stopped_prev = False

            for stopped_label in vehicle_df["stopped"].tolist():
                # convert bool string to boolean type
                bool_stopped_current = True if stopped_label == "true" else False

                if bool_stopped_current != bool_stopped_prev:
                    # going from moving to stopped
                    if bool_stopped_current:
                        stop_counts.append(vehicle_type)
                    # going from stopped to moving
                    elif not bool_stopped_current:
                        start_counts.append(vehicle_type)

                    bool_stopped_prev = bool_stopped_current

        # organize into df
        stops_dict, starts_dict = (
            collections.Counter(stop_counts),
            collections.Counter(start_counts),
        )
        stops_df = pd.DataFrame.from_dict(stops_dict, orient="index", columns=["stops"])
        starts_df = pd.DataFrame.from_dict(
            starts_dict, orient="index", columns=["starts"]
        )

        stops_df.index.name = "vehicle_type"
        stops_df.reset_index(inplace=True)

        starts_df.index.name = "vehicle_type"
        starts_df.reset_index(inplace=True)

        return stops_df.merge(starts_df, how="outer", on="vehicle_type")

    def get_true_parked_counts(self) -> pd.DataFrame:
        """Computes the number of parked vehicles from the ground_truth_df. 
        """
        parked_counter = []
        for vehicle_id, vehicle_df in self.ground_truth_df.groupby("vehicle_id"):
            parked_status = vehicle_df["parked"].tolist()[0]
            # convert str to bool
            parked_boolean = False if parked_status == "false" else True
            vehicle_type = vehicle_df["vehicle_type"].tolist()[0]

            if parked_boolean:
                parked_counter.append(vehicle_type)

        parked_dict = collections.Counter(parked_counter)
        parked_df = pd.DataFrame.from_dict(
            parked_dict, orient="index", columns=["parked"]
        )
        parked_df.index.name = "vehicle_type"
        parked_df.reset_index(inplace=True)

        return parked_df
