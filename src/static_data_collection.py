from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths
from traffic_analysis.d01_data.collect_video_data import collect_camera_videos, upload_videos
from multiprocessing import Process

params = load_parameters()
creds = load_credentials()
paths = load_paths()

blob_credentials = creds[paths['blob_creds']]


def collect_camera_videos_fn():
    collect_camera_videos(download_url=params['jamcam_url'],
                          blob_credentials=blob_credentials,
                          delay=params["delay"])


def upload_videos_fn():
    upload_videos(blob_credentials=blob_credentials,
                  delay=1)


def run_in_parallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


print("Running download and upload functions in parallel")
run_in_parallel(collect_camera_videos_fn, upload_videos_fn)

