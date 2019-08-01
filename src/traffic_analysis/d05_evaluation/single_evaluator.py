import pandas as pd

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name


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
