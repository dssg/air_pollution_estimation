from ..d00_utils.data_retrieval import get_single_video_s3_to_np
from ..d04_modelling.classify_objects import classify_objects
from os import path
import numpy as np
import pandas as pd
import pickle as
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


def yolo_output_df(obj_bounds, obj_labels, obj_label_confidences, camera_id, date, time):
    """Formats the output of yolo on one video. Returns as pandas df. 

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

    date = datetime.datetime.strptime(date, '%Y%m%d').date()
    time = datetime.datetime.strptime(time, '%H%M').time()

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
    yolo_df = yolo_df[["camera_id", "frame_id", "date", "time", "obj_bounds", "obj_classification", "confidence"]]
    return yolo_df


def yolo_report_stats(yolo_df):
    '''Report summary statistics for the output of YOLO on one video. 

    Keyword arguments: 
    yolo_df -- pandas df containing formatted output of YOLO for one video (takes the output of yolo_output_df())

    Returns: 
    obj_counts_frame: counts of various types of objects per frame
    video_summary: summary statistics over whole video 


    '''
    obj_counts_frame=yolo_df.groupby(["frame_id", "obj_classification"]).size().reset_index(name = 'obj_count')

    #long to wide format
    #some object types were not detected in a frame, so we fill these NAs with 0s
    obj_counts_frame=obj_counts_frame.pivot(index='frame_id', columns='obj_classification', values='obj_count').fillna(value = 0)
    
    #get the sum of each type of object over all frames
    sums = obj_counts_frame.aggregate(func = "sum")

    #drop count because it is just the number of rows in the df
    video_summary = obj_counts_frame.describe().drop(["count", "25%", "75%"], axis = 0)
    video_summary.loc['sum'] = sums

    return obj_counts_frame, video_summary


if __name__ == '__main__':
    #example of how to interface functions above with yolo code 
    local_mp4_path=path.abspath(path.join(basepath,"..", "..", "data/sample_video_data/testvid.mp4"))
    pkl_path=path.abspath(path.join(basepath,"..", "..", "data/pickled/", "yolo_res"))
    save_path = path.abspath(path.join(basepath,"..", "..", "data/sample_yolo_output", "sample_yolo_output.csv"))

    #sample input 
    camera="06508"
    date="20190603" #yyyymmdd
    time="1145"

    os.chdir(".")
    with open('../../conf/base/parameters.yml') as f:
       params = yaml.safe_load(f)['modelling']
    os.chdir(".")
    with open('../../conf/base/paths.yml') as f:
       paths = yaml.safe_load(f)['paths']
    
    #1. load video from s3, run csvlib YOLO, dump output as PKL to decrease development time

    retrieve_single_video(camera=camera, date=date, time=time, paths=paths)
    # s3_to_local_mp4(camera=camera, date=date, time=time,
    #                 local_mp4_path=local_mp4_path)
    obj_bounds, obj_labels, obj_label_confidences=classify_objects(local_mp4_path=local_mp4_path, params = params)

    with open(pkl_path, 'wb') as fh:
        pkl.dump([obj_bounds, obj_labels, obj_label_confidences], fh)

    # 2. load output from pkl file, process and write as csv
    with open(pkl_path, 'rb') as fh:
            yolo_res = pkl.load(fh)

    obj_bounds, obj_labels, obj_label_confidences = yolo_res

    yolo_df=yolo_output_df(obj_bounds, obj_labels, obj_label_confidences, camera, date, time)

    print(yolo_df.columns.tolist())
    yolo_df.to_csv(save_path)

    # 3. Reload csv, generate stats: group on frame,  vehicle type, number

    yolo_df = pd.read_csv(save_path)
    obj_counts_frame, video_summary = yolo_report_stats(yolo_df)
    print(obj_counts_frame.head(), video_summary.head())
