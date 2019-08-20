from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths
from traffic_analysis.d01_data.collect_video_data import collect_camera_videos

params = load_parameters()
creds = load_credentials()
paths = load_paths()

tfl_camera_api = params['tfl_camera_api']
s3_credentials = creds[paths['s3_creds']]

collect_camera_videos(download_url=params['jamcam_url'],
                      s3_credentials=s3_credentials,
                      iterations=1,
                      delay=0)
