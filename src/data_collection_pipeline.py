from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths
from traffic_analysis.d01_data.collect_video_data import download_camera_meta_data, collect_camera_videos, upload_videos, rename_videos
from multiprocessing import Process

params = load_parameters()
paths = load_paths()
tfl_cam_api = params['tfl_cam_api']
cam_file = paths['cam_file']
iterations = params['iterations']
delay = params['delay']
local_video_dir = paths['temp_video']

# download camera data from tfl
download_camera_meta_data(tfl_camera_api=tfl_cam_api,
                          camera_meta_data_file=cam_file)
print("Downloaded tfl camera details.")


def collect_camera_videos_fn():
    collect_camera_videos(local_video_dir=local_video_dir,
                          download_url=params['jamcam_url'],
                          cam_file=cam_file,
                          iterations=iterations,
                          delay=delay)


def upload_videos_fn():
    upload_videos(local_video_dir=local_video_dir,
                  credentials=paths,
                  iterations=iterations,
                  delay=delay)


def rename_videos_fn():
    rename_videos(paths=paths,
                  params=params,
                  chunk_size=1000)


def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


print("Running download and upload functions in parallel")
runInParallel(collect_camera_videos_fn, upload_videos_fn, rename_videos_fn)
