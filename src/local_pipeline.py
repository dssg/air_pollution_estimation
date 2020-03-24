import os
import glob
import pandas as pd

from traffic_analysis.d00_utils.load_confs import (load_credentials,
                                                   load_parameters, load_paths)
from traffic_analysis.d00_utils.download_yolo_local import download_yolo_local
from traffic_analysis.d04_modelling.tracking.tracking_analyser import \
    TrackingAnalyser
from traffic_analysis.d03_processing.update_frame_level_local import \
    update_frame_level_local
from traffic_analysis.d03_processing.update_video_level_local import \
    update_video_level_local


params = load_parameters()
paths = load_paths()
creds = load_credentials()
blob_credentials = creds[paths['blob_creds']]

download_yolo_local(paths['local_detection_model'])

chunk_size = params['chunk_size']
make_video = params['make_video']

selected_videos = glob.glob(paths['local_video'] + '/*mp4')
analyser = TrackingAnalyser(params=params, paths=paths, blob_credentials=blob_credentials)
dfs = []

# select chunks of videos and classify objects
while selected_videos:
    file_names = selected_videos[:chunk_size]
    success, frame_level_df, runtime_list, lost_tracking = update_frame_level_local(analyser=analyser,
                                              file_names=file_names,
                                              paths=paths,
                                              creds=creds,
                                              make_video=make_video)

    video_level_df = update_video_level_local(analyser=analyser,
                             frame_level_df=frame_level_df)

    dfs.append(video_level_df)

    # move processed videos to processed folder
    for file in file_names:
        os.rename(file, paths['local_processed_video'] + file.split('/')[-1])

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]

df_to_save = pd.concat(dfs, ignore_index=True)
df_to_save.to_csv(paths['local_results'] + 'output.csv')

