
from src.d00_utils.data_retrieval import describe_s3_bucket, retrieve_single_video
from src.d00_utils.load_confs import load_parameters, load_paths

from src.d04_modelling.classify_objects import classify_objects

params = load_parameters()
paths = load_paths()

#describe_s3_bucket(paths)
camera = '03675'
date = '2019-06-20'
time = '13:25:31.321785'
video = retrieve_single_video(camera, date, time, paths, bool_keep_data=True)
local_mp4_path = paths['local_video'] + date + "_" + time + "_" + camera + '.mp4'
obj_bounds, obj_labels, obj_label_confidences = classify_objects(local_mp4_path, params, vid_time_length=10, make_video=True)
print(obj_bounds)
print(obj_labels)
print(obj_label_confidences)