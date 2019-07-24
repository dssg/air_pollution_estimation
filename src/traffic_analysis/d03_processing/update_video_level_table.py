import datetime

from traffic_analysis.d00_utils.data_access import db
from traffic_analysis.d03_processing.add_to_table_sql import add_to_table_sql
from traffic_analysis.d05_reporting.report_yolo import yolo_report_stats


def update_video_level_table(analyzer, file_names, paths, creds):
    """ Update the video level table on the database based on the videos in the files list
                Args:
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    creds (dict): dictionary of credentials from yml file

                Returns:

    """
    # Build the sql string
    datetimes = []
    camera_ids = []
    for file in file_names:

        name = file.split('/')[-1]
        datetimes.append(datetime.datetime.strptime(name.split('_')[0], "%Y-%m-%d %H:%M:%S.%f"))
        camera_ids.append(name.split('_')[-1][:-4])

    filter_string = ''
    for i in range(len(file_names)):
        filter_string += '(camera_id=\'' + camera_ids[i] + '\' AND video_upload_datetime=\'' + str(datetimes[i]) + '\') OR '

    filter_string = filter_string[:-4]
    sql_string = "SELECT * FROM frame_stats WHERE %s;" % (filter_string)

    # Get the data from the database as a df
    db_host = paths['db_host']
    db_name = paths['db_name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn_string = "password=%s user=%s dbname=%s host=%s" % (
        db_pass, db_user, db_name, db_host)

    db_obj = db(conn_string=conn_string)
    frame_level_df = db_obj.execute_raw_query(sql=sql_string)
    bboxes = []
    for x, y, w, h in zip(frame_level_df['bbox_x'].values, frame_level_df['bbox_y'].values, frame_level_df['bbox_w'].values, frame_level_df['bbox_h'].values):
        bboxes.append([x, y, w, h])
    frame_level_df['bboxes'] = bboxes
    frame_level_df.drop('bbox_x', axis=1, inplace=True)
    frame_level_df.drop('bbox_y', axis=1, inplace=True)
    frame_level_df.drop('bbox_w', axis=1, inplace=True)
    frame_level_df.drop('bbox_h', axis=1, inplace=True)
    print(frame_level_df.head(3))

    # Create video level table and add to database
    video_level_df = analyzer.construct_video_level_df(frame_level_df)

    add_to_table_sql(df=video_level_df,
                     table='video_stats',
                     creds=creds,
                     paths=paths)

    return
