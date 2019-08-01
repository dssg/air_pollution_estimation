import collections
import numpy as np
import pandas as pd
import collections

from traffic_analysis.d05_evaluation.single_evaluator import SingleEvaluator


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
