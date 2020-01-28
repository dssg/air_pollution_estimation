from traffic_analysis.d01_data.collect_video_data import upload_videos
from traffic_analysis.d00_utils.load_confs import load_parameters, load_credentials, load_paths

params = load_parameters()
creds = load_credentials()
paths = load_paths()

blob_credentials = creds[paths['blob_creds']]

upload_videos(blob_credentials=blob_credentials,
              iterations=1,
              delay=0)
