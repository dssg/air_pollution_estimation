# from ..d02_intermediate.view_jam_cams import s3_to_local_mp4, classify_objects
from os import path
import numpy as np
import pandas as pd
import pickle as pkl
import datetime
basepath=path.dirname(__file__) #path of current script


def frame_info_to_df(obj_info_aggregated, frame_ind, camera_id, date,time):
    """Parse the info corresponding to one frame into one pandas df

    Keyword arguments: 
    obj_info_aggregated -- np array, contains 3 subarrays: object bounds, object
                           labels, object label confidences
    frame_ind -- np arrays of lists
    camera_id -- the camera which the frame came from (int)
    date -- date of the video which the frame came from (Python datetime date object)
    time -- time of the video which the frame came from (Python datetime time object)

    """
    frame_df = pd.DataFrame(obj_info_aggregated, columns = ['obj_bounds', 'obj_classification', 'confidence'])
    frame_df["frame_id"] = frame_ind
    frame_df["camera_id"] = camera_id
    frame_df["date"] = date
    frame_df["time"] = time

    return frame_df


def process_datetime(date, time): 
    """ Converts string date times into Python datetime objects

    Keyword arguments: 
    date -- string date, format yyyymmdd
    time -- string time, format hhmm
    """

    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
    hour, minute = int(time[:2]), int(time[2:])

    date_typed = datetime.date(year = year, month = month, day = day)
    time_typed = datetime.time(hour = hour, minute = minute, second = 0)

    return date_typed, time_typed


def yolo_output_df(obj_bounds, obj_labels, obj_label_confidences, camera_id, date, time):
    """Formats the output of yolo 

    Keyword arguments: 
    obj_bounds -- nested list (top level is frames, next level is objs detected
                  in each frame)
    obj_labels -- nested list, same structure as above
    obj_label_confidences -- nested list, same structure as above
    camera_id -- the camera id which the frame came from (string)
    date -- string date, format yyyymmdd
    time -- string time, format hhmm

    """
    obj_bounds, obj_labels, obj_label_confidences=np.array(obj_bounds), np.array(obj_labels), np.array(obj_label_confidences)

    #ensure all three lists have same number of frames (one entry in list corresp to one frame)
    num_frames = obj_bounds.shape[0]
    assert obj_labels.shape[0] == num_frames 
    assert obj_label_confidences.shape[0] == num_frames 

    date,time = process_datetime(date, time)

    frame_df_list = []

    #loop over frames 
    for frame_ind in range(num_frames):
        obj_bounds_np = [np.array(bound) for bound in obj_bounds[frame_ind]]

        obj_info_aggregated = np.array([obj_bounds_np, obj_labels[frame_ind], 
                                        obj_label_confidences[frame_ind]]).transpose()

        frame_df = frame_info_to_df(obj_info_aggregated, frame_ind, int(camera_id), date,time)
        frame_df_list.append(frame_df)

    yolo_df = pd.concat(frame_df_list)

    #yolo_df index is the index of an objected detected over a frame
    yolo_df.index.name = "obj_ind"
    return yolo_df



if __name__ == '__main__':
    #example of how to interface functions above with yolo code 
    local_mp4_path=path.abspath(path.join(basepath,"..", "..", "data/sample_video_data/testvid.mp4"))
    pkl_path=path.abspath(path.join(basepath,"..", "..", "data/pickled/", "yolo_res"))
    save_path = path.abspath(path.join(basepath,"..", "..", "data/sample_yolo_output", "sample_yolo_output.csv"))

    camera="06508"
    date="20190603"
    time="1145"
    extension=".mp4"

    # #need to test this but don't really need this? 
    # s3_to_local_mp4(camera=camera, date=date, time=time,
    #                 extension=extension, local_mp4_path=local_mp4_path)
    # obj_bounds, obj_labels, obj_label_confidences=classify_objects(local_mp4_path=local_mp4_path)

    # with open(pkl_path, 'wb') as fh:
    #     pkl.dump([obj_bounds, obj_labels, obj_label_confidences], fh)


    with open(pkl_path, 'rb') as fh:
            yolo_res = pkl.load(fh)

    obj_bounds, obj_labels, obj_label_confidences = yolo_res

    yolo_df=yolo_output_df(obj_bounds, obj_labels, obj_label_confidences, camera, date, time)

    print(yolo_df.columns.tolist())
    yolo_df.to_csv(save_path)