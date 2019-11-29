import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.load_confs import (load_credentials,
                                                   load_parameters, load_paths)
from traffic_analysis.d02_ref.ref_utils import generate_dates

from_date = "2019-11-29"
to_date = "2019-11-29"

print('From: ' + from_date + ' To: ' + to_date)

to_date = dateutil.parser.parse(to_date).date()
from_date = dateutil.parser.parse(from_date).date()

results_dict = {}

params = load_parameters()
paths = load_paths()
creds = load_credentials()

dl = DataLoaderSQL(creds=creds, paths=paths)

# Generate the list of dates
dates = generate_dates(from_date, to_date)
for date in dates:
    date = date.strftime('%Y-%m-%d')
    sql = "select * from video_stats where DATE(video_upload_datetime) = \'" + date + "\'"

    df = dl.select_from_table(sql=sql)

    duplicates = df.duplicated()
    print('Number of duplicate rows for date: ' + date + ' is ' + str(sum(duplicates.tolist())))

    video_wise_df = df.drop_duplicates(subset=['camera_id', 'video_upload_datetime'])

    # Number of videos per camera for that day
    plt.figure(figsize=(12, 10))
    camera_ids, counts = np.unique(video_wise_df['camera_id'].values, return_counts=True)
    plt.bar(np.arange(camera_ids.shape[0]), counts)
    plt.xticks(ticks=np.arange(camera_ids.shape[0]), labels=camera_ids, rotation=90)
    plt.suptitle('Number of videos for each camera on ' + date)
    plt.savefig('video_count_per_camera_' + date + '.png')
    plt.close()

    # Number of videos per hour for that day
    max_num_videos = camera_ids.shape[0] * (60 / 4)
    date_time_df = video_wise_df['video_upload_datetime'].to_frame()
    date_time_df.set_index('video_upload_datetime', drop=False, inplace=True)
    date_time_df.groupby(pd.TimeGrouper(freq='60Min')).count().plot(kind='bar', figsize=(12, 10))
    plt.axhline(y=max_num_videos, xmin=0, xmax=1, linestyle='--', color='r')
    plt.suptitle('Number of videos over time for ' + date)
    legend = plt.legend()
    legend.remove()
    plt.savefig('video_count_over_time_' + date + '.png')
    plt.close()

    num_cols = 5
    num_rows = int(np.ceil(camera_ids.shape[0] / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))

    for camera_id, axis in zip(camera_ids, axes.reshape(-1)):

        camera_id_df = df.loc[df['camera_id'] == camera_id]
        types = np.unique(df['vehicle_type'].values)

        for type in types:

            camera_id_type_df = camera_id_df.loc[camera_id_df['vehicle_type'] == type]
            camera_id_type_df = camera_id_type_df.sort_values(by=['video_upload_datetime'])

            axis.plot(camera_id_type_df['video_upload_datetime'].values,
                    camera_id_type_df['counts'], '--o', label=type)

        # Set title and labels for axes
        axis.legend()
        axis.set(xlabel="Date",
               ylabel="Vehicle Counts",
               title="Counts for Camera " + camera_id)
        for tick in axis.get_xticklabels():
            tick.set_rotation(45)

    plt.tight_layout()
    plt.legend()
    fig.savefig("counts_" + date + ".png")
