
from src.d00_utils.data_retrieval import retrieve_daterange_videos_s3_to_np, \
    load_videos_from_local
from src.d00_utils.load_confs import load_parameters, load_paths

from src.d04_modelling.classify_objects import classify_objects

params = load_parameters()
paths = load_paths()

"""
camera = '03675'
date = '2019-06-20'
time = '13:25:31.321785'
video = retrieve_single_video(camera, date, time, paths, bool_keep_data=True)
"""

#retrieve_daterange_videos_s3_to_np(paths, from_date='2019-06-20', to_date='2019-06-20', bool_keep_data=True)
videos, names = load_videos_from_local(paths)
obj_bounds, obj_labels, obj_label_confidences = classify_objects(videos, params, paths,
                                                                 vid_time_length=10, make_videos=False,
                                                                 vid_names=names)
print(obj_bounds)
print(obj_labels)
print(obj_label_confidences)