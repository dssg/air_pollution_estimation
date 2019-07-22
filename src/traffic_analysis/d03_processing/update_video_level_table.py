import datetime

from traffic_analysis.d00_utils.data_access import db
from traffic_analysis.d05_reporting.report_yolo import yolo_report_stats


def update_video_level_table(file_names, paths, params, creds):
    """ Update the frame level table on s3 (pq) based on the videos in the files list
                Args:
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    params (dict): dictionary of parameters from yml file

                Returns:

    """

    # Get the data from the database as a df

    datetimes = []
    camera_ids = []

    # Build the sql string
    for file in file_names:

        name = file.split('/')[-1]
        datetimes.append(datetime.datetime.strptime(name.split('_')[0], "%Y-%m-%d %H:%M:%S.%f"))
        camera_ids.append(name.split('_')[-1][:-4])

    filter_string = ''

    for i in range(len(file_names)):

        filter_string += '(camera_id=' + camera_ids[i] + ' AND datetime=' + str(datetimes[i]) + ') OR '

    filter_string = filter_string[:-4]
    print(filter_string)

    sql_string = "SELECT * FROM frame_stats WHERE %s;" % (filter_string)

    db_host = paths['db_host']
    db_name = paths['db_name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn_string = "password=%s user=%s dbname=%s host=%s" % (
        db_pass, db_user, db_name, db_host)

    db_obj = db(conn_string=conn_string)
    result = db_obj.execute_raw_query(sql=sql_string)

    return
