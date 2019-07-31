import numpy as np
import pandas as pd
import xml.etree.ElementTree as ElementTree


def parse_annotations(xml_files, paths, bool_print_summary=False):
    """ Parse the XML files containing the manual annotations
                Args:
                    xml_files: list of the XML files to parse
                    paths: dictionary of paths (needs 'annotations' path)
                    bool_print_summary: boolean for printing summaries of each parsed XML file
                Returns:
                    pandas dataframe containing the annotations
                Raises:
    """

    for xml_file in xml_files:
        root = ElementTree.parse(paths["annotations"] + xml_file + ".xml").getroot()
        annotated_results = {
            "id": [],
            "frame": [],
            "occluded": [],
            "bounds": [],
            "type": [],
            "parked": [],
            "stopped": [],
        }

        for track in root.iter("track"):
            if track.attrib["label"] == "vehicle":
                for frame in track.iter("box"):
                    annotated_results["id"].append(int(track.attrib["id"]))
                    annotated_results["frame"].append(int(frame.attrib["frame"]))
                    annotated_results["occluded"].append(int(frame.attrib["occluded"]))
                    annotated_results["bounds"].append(
                        [
                            float(frame.attrib["xtl"]),
                            float(frame.attrib["ytl"]),
                            float(frame.attrib["xbr"]),
                            float(frame.attrib["ybr"]),
                        ]
                    )
                    for attr in frame.iter("attribute"):
                        annotated_results[attr.attrib["name"]].append(attr.text)

        df = pd.DataFrame.from_dict(annotated_results)

        if bool_print_summary:
            print("Number of vehicles:")
            print(df.groupby("type")["id"].nunique())
            print("Parked vehicles:")
            print(df.groupby("id")["parked"].unique())
            stops_df = get_stop_counts(df)
            print("Number of Stops:")
            print(stops_df)

    return df


def get_stop_counts(annotations_df):
    """ Get the number of stops for each vehicle
                    Args:
                        annotations_df: pandas dataframe containing the annotations
                    Returns:
                        pandas dataframe containing the vehicle ids and the number of stops
                    Raises:
    """
    ids, counts = [], []
    df_grouped = annotations_df.sort_values(["frame"], ascending=True).groupby("id")
    for group in df_grouped:
        bool_stopped = False
        num_stops = 0
        for val in group[1]["stopped"].tolist():
            if val == "true" and not bool_stopped:
                bool_stopped = True
            elif val == "false" and bool_stopped:
                num_stops += 1
                bool_stopped = False
        if bool_stopped:
            num_stops += 1
        ids.append(group[1]["id"].tolist()[0])
        counts.append(num_stops)
    stops_df = pd.DataFrame(
        data=np.array(list(zip(ids, counts))), columns=["id", "num_stops"]
    )
    return stops_df
