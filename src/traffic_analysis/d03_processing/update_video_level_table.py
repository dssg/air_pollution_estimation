import datetime

from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d05_reporting.report_yolo import yolo_report_stats


def update_video_level_table(file_names, paths, creds):
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
    for filename in file_names:
        name = filename.split('/')[-1]
        camera_id, date_time = parse_video_or_annotation_name(name)
        datetimes.append(date_time)
        camera_ids.append(camera_id)

    filter_string = ''
    for i in range(len(file_names)):
        filter_string += '(camera_id=\'' + camera_ids[i] + '\' AND datetime=\'' + str(datetimes[i]) + '\') OR '

    filter_string = filter_string[:-4]
    sql_string = "SELECT * FROM frame_stats WHERE %s;" % (filter_string)

    # Get the data from the database as a df
    db_host = paths['db_host']
    db_name = paths['db_name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn_string = "password=%s user=%s dbname=%s host=%s" % (
        db_pass, db_user, db_name, db_host)

    db_obj = DataLoaderSQL(creds=creds, paths=paths)
    frame_level_df = db_obj.execute_raw_sql_query(sql=sql_string)

    # Create video level table and add to database
    video_level_df = yolo_report_stats(frame_level_df)

    db_obj.add_to_sql(df=video_level_df, table_name='video_stats')

    return
