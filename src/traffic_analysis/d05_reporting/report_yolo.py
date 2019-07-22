import numpy as np
import pandas as pd
import dateutil.parser
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
import numpy as np
import pandas as pd


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
    frame_df["datetime"] = date_time

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
        obj_bounds = np.array(values['bounds'])
        obj_labels = np.array(values['labels'])
        obj_label_confidences = np.array(values['confidences'])

        # ensure all three lists have same number of frames (one entry in list corresp to one frame)
        num_frames = obj_bounds.shape[0]
        assert obj_labels.shape[0] == num_frames
        assert obj_label_confidences.shape[0] == num_frames

        filename = name.split("_")
        time_obj = filename[1].replace("-",":") if len(filename) > 2 else " "
        datetimestring = "%s %s"%(filename[0], time_obj)
        datetimestring = datetimestring.strip()
        print(datetimestring)
        date_time = dateutil.parser.parse(datetimestring)
        camera_id = filename[-1][:-4]
        camera_id, date_time = parse_video_or_annotation_name(name)

        frame_df_list = []

        # loop over frames
        for frame_ind in range(num_frames):
            obj_bounds_np = [np.array(bound)
                             for bound in obj_bounds[frame_ind]]

            obj_info_aggregated = np.array([obj_bounds_np, obj_labels[frame_ind],
                                            obj_label_confidences[frame_ind]]).transpose()

            frame_df = frame_info_to_df(obj_info_aggregated, frame_ind, camera_id, date_time)
            frame_df = frame_info_to_df(
                obj_info_aggregated, frame_ind, camera_id, date_time)
            frame_df_list.append(frame_df)

        yolo_df = pd.concat(frame_df_list)

        # yolo_df index is the index of an objected detected over a frame
        yolo_df.index.name = "obj_ind"
        yolo_df = yolo_df[["camera_id", "frame_id", "datetime",
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


def yolo_report_stats(yolo_df):
    '''Report summary statistics for the output of YOLO on one video. 

    Keyword arguments: 
    yolo_df -- pandas df containing formatted output of YOLO for one video (takes the output of yolo_output_df())

    Returns: 
    obj_counts_frame: counts of various types of objects per frame
    video_summary: summary statistics over whole video 


    '''
    dfs = []
    if yolo_df.empty:
        return pd.DataFrame()
    grouped = yolo_df.groupby(['camera_id', 'datetime'])

    for name, group in grouped:
        obj_counts_frame=group.groupby(["frame_id", "obj_classification"]).size().reset_index(name = 'obj_count')
        
        for type in obj_counts_frame['obj_classification'].unique():
            type_df = obj_counts_frame[obj_counts_frame['obj_classification']==type]
            count_df = pd.DataFrame([type_df['obj_count'].mean()], columns=['counts'])
            count_df['vehicle_type'] = type
            count_df['camera_id'] = group['camera_id'].iloc[0]
            count_df['datetime'] = group['datetime'].iloc[0]
            count_df['starts'] = 0
            count_df['stops'] = 0
            assert group['datetime'].nunique() == 1, "Non-unique datetime..." + str(group['datetime'].unique())
            assert group['camera_id'].nunique() == 1, "Non-unique camera_id"
            dfs.append(count_df)

    df = pd.concat(dfs).fillna(0)
    return df
