import datetime
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d05_reporting.report_yolo import yolo_report_stats


def update_video_level_table(frame_level_df=None, file_names=None, paths=None, creds=None, params=None):
    """ Update the video level table on the database based on the videos in the files list
                Args:
                    frame_level_df (dataframe): dataframe containing the frame level stats, if none then
                    this is loaded from the database using the file names
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    creds (dict): dictionary of credentials from yml file

                Returns:

    """
    db_obj = DataLoaderSQL(creds=creds, paths=paths)

    if(frame_level_df is None):
        # Build the sql string
        filter_string = ''

        for filename in file_names:
            name = filename.split('/')[-1]
            camera_id, date_time = parse_video_or_annotation_name(name)
            filter_string += '(camera_id=\'' + camera_id + '\' AND video_upload_datetime=\'' + str(date_time) + '\') OR '

        filter_string = filter_string[:-4]
        sql_string = "SELECT * FROM frame_stats WHERE %s;" % (filter_string)
        frame_level_df = db_obj.execute_raw_sql_query(sql=sql_string)

    # Create video level table and add to database
    video_level_df = yolo_report_stats(frame_level_df=frame_level_df, params=params)
    video_level_df['creation_datetime'] = datetime.datetime.now()

    db_obj.add_to_sql(df=video_level_df, table_name='video_stats')

    return
