from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths
from traffic_analysis.d01_data.collect_video_data import download_camera_meta_data, collect_camera_videos

params = load_parameters()
creds = load_credentials()
paths = load_paths()

tfl_camera_api = params['tfl_camera_api']
s3_credentials = creds[paths['s3_creds']]

# download camera data from tfl
download_camera_meta_data(tfl_camera_api=tfl_camera_api,
                          s3_credentials=s3_credentials)
print("Downloaded tfl camera details.")


collect_camera_videos(download_url=params['jamcam_url'],
                      s3_credentials=s3_credentials,
                      iterations=1,
                      delay=0)
