import time as Time
import datetime

from src.d00_utils.load_confs import load_parameters, load_paths
from src.d00_utils.data_retrieval import retrieve_videos_s3_to_np

if __name__ == "__main__":
    params = load_parameters()
    paths = load_paths()

    cameras = params['tims_camera_list'][:2]
    from_date = '2019-06-20'
    to_date = str(datetime.datetime.now())[:10]

    time = datetime.datetime(100, 1, 1, hour=0, minute=0, second=0)
    dt_hours = 3

    while(time < datetime.datetime(100, 1, 1, hour=23, minute=59, second=59)):

        from_time = (time - datetime.timedelta(minutes=2)).strftime("%H-%M-%S")
        to_time = (time + datetime.timedelta(minutes=2)).strftime("%H-%M-%S")

        print('Collecting Videos for ' + (from_time) + ' to ' + str(to_time))
        t = Time.time()
        videos, names = retrieve_videos_s3_to_np(paths, from_date=from_date, to_date=to_date,
                                                 from_time=from_time, to_time=to_time,
                                                 camera_list=cameras,
                                                 bool_keep_data=True)
        print('Collected videos in ' + str(Time.time() - t) + ' seconds')
        time += datetime.timedelta(hours=dt_hours)
