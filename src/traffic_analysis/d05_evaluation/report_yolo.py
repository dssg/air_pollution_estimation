import numpy as np
import pandas as pd
import dateutil.parser
import numpy as np
import pandas as pd

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
# TODO Remove this code as we no longer take a YOLO only approach

def frame_info_to_df(obj_info_aggregated, frame_ind, camera_id, date_time):
    """Parse the info corresponding to one frame into one pandas df

    Keyword arguments: 
    obj_info_aggregated -- np array, contains 3 subarrays: object bounds, object
                           labels, object label confidences
    frame_ind -- np arrays of lists
    camera_id -- the camera which the frame came from (int)
    date -- date of the video which the frame came from (Python datetime date object)
    time -- time of the video which the frame came from (Python datetime time object)

    """
    frame_df = pd.DataFrame(obj_info_aggregated, columns=[
                            'obj_bounds', 'obj_classification', 'confidence'])
    frame_df["frame_id"] = frame_ind
    frame_df["camera_id"] = camera_id
    frame_df["video_upload_datetime"] = date_time

    return frame_df


def yolo_output_df(yolo_dict):
    """Formats the output of yolo on one video. Returns as pandas df. 

    Keyword arguments: 
        yolo_dict (dict): nested dictionary where each video is a key for a dict containing:
                obj_bounds (list of np arrays): n-dim list of list of arrays marking the corners of the bounding boxes of objects, for n frames
                obj_labels (list of str): n-dim list of list of labels assigned to classified objects, for n frames
                obj_label_confidences (list of floats): n-dim list of list of floats denoting yolo confidences, for n frames

    Returns:
        df (df): pandas dataframe containing all the values from yolo_dict

    """
    df_list = []

    for video_num, (name, values) in enumerate(yolo_dict.items()):
        obj_bounds = np.array(values["bounds"])
        obj_labels = np.array(values["labels"])
        obj_label_confidences = np.array(values["confidences"])

        # ensure all three lists have same number of frames (one entry in list corresp to one frame)
        num_frames = obj_bounds.shape[0]
        assert obj_labels.shape[0] == num_frames
        assert obj_label_confidences.shape[0] == num_frames

        camera_id, date_time = parse_video_or_annotation_name(name)

        frame_df_list = []

        # loop over frames
        for frame_ind in range(num_frames):
            obj_bounds_np = [np.array(bound) for bound in obj_bounds[frame_ind]]

            obj_info_aggregated = np.array(
                [obj_bounds_np, obj_labels[frame_ind], obj_label_confidences[frame_ind]]
            ).transpose()

            frame_df = frame_info_to_df(
                obj_info_aggregated, frame_ind, camera_id, date_time)
            frame_df_list.append(frame_df)

        yolo_df = pd.concat(frame_df_list)

        # yolo_df index is the index of an objected detected over a frame
        yolo_df.index.name = "obj_ind"
        yolo_df = yolo_df[["camera_id", "frame_id", "video_upload_datetime",
                           "obj_bounds", "obj_classification", "confidence"]]
        df_list.append(yolo_df)
    df = pd.DataFrame()
    # Concatenate dataframes
    if df_list:
        df = pd.concat(df_list)

    x, y, w, h = [], [], [], []
    for vals in df['obj_bounds'].values:
        x.append(vals[0])
        y.append(vals[1])
        w.append(vals[2])
        h.append(vals[3])
    df['box_x'] = x
    df['box_y'] = y
    df['box_w'] = w
    df['box_h'] = h
    df.drop('obj_bounds', axis=1, inplace=True)

    return df


def yolo_report_stats(frame_level_df, params):
    '''Report summary statistics for the output of YOLO on one video. 

    Keyword arguments: 
    yolo_df -- pandas df containing formatted output of YOLO for one video (takes the output of yolo_output_df())

    Returns: 
    obj_counts_frame: counts of various types of objects per frame
    video_summary: summary statistics over whole video 


    '''

    # get frame level object counts
    frame_object_type = (frame_level_df
                         .rename(columns={'datetime': 'video_upload_datetime'})
                         .groupby(['camera_id', 'video_upload_datetime', "frame_id", "obj_classification"])
                         .confidence
                         .count()
                         .reset_index()
                         .rename(columns={'confidence': 'obj_count'}))

    # get video level estimates
    video_level_counts = (frame_object_type
                          .groupby(['camera_id', 'video_upload_datetime', 'obj_classification'])
                          .obj_count
                          .mean()
                          .reset_index())

    # restrict to objects of interest and impute 0 for non-detected items
    all_vehicle_types = pd.DataFrame({'obj_classification': params['selected_labels']})
    all_video_dates = frame_object_type[['camera_id', 'video_upload_datetime']].drop_duplicates()

    video_level_df = (pd.merge(left=all_vehicle_types.assign(foo=1),
                               right=all_video_dates.assign(foo=1),
                               on=['foo'],
                               how='left')
                      .drop(columns=['foo']))

    video_level_df = pd.merge(left=video_level_df,
                              right=video_level_counts,
                              on=['camera_id', 'video_upload_datetime', 'obj_classification'],
                              how='left')
    video_level_df = video_level_df.fillna(0)

    return video_level_df
