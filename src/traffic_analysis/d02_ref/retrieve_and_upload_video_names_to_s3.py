import datetime
import time as Time
import dateutil.parser

from traffic_analysis.d02_ref.ref_utils import upload_json_to_s3
from traffic_analysis.d02_ref.ref_utils import generate_dates
from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3


def retrieve_and_upload_video_names_to_s3(ouput_file_name,
                                          paths,
                                          from_date='2019-06-01',
                                          to_date=str(
                                              datetime.datetime.now().date()),
                                          from_time='00-00-00',
                                          to_time='23-59-59',
                                          camera_list=None,
                                          return_files_flag=False):
    """Upload a json to s3 containing the filepaths for videos between the dates, times and cameras specified.

        Args:
            ouput_file_name (str): name of the json to be saved
            paths (dict): dictionary containing temp_video, raw_video, s3_profile and bucket_name paths
            from_date (str): start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
            to_date (str): end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
            from_time (str): start time for retrieving videos, if None then will retrieve from the start of the day
            to_time (str): end time for retrieving videos, if None then will retrieve up to the end of the day
            camera_list (list): list of cameras to retrieve from, if None then retrieve from all cameras
        Returns:

    """
    print('From: ' + from_date + ' To: ' + to_date)
    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    s3_video = paths['s3_video']
    to_date = dateutil.parser.parse(to_date).date()
    from_date = dateutil.parser.parse(from_date).date()
    from_time = dateutil.parser.parse(format_time(from_time)).time()
    to_time = dateutil.parser.parse(format_time(to_time)).time()
    selected_files = []

    # Generate the list of dates
    dates = generate_dates(from_date, to_date)
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        prefix = "%s%s/" % (s3_video, date)

        # fetch video filenames
        elapsed_time, files = get_names_of_folder_content_from_s3(
            bucket_name, prefix, s3_profile)
        print('Extracting {} file names for date {} took {} seconds'.format(len(files),
                                                                            date,
                                                                            elapsed_time))
        if not files:
            continue

        for filename in files:
            if filename:
                res = filename.split('_')
                camera_id = res[-1][:-4]
                time_of_day = res[0].split(".")[0]
                time_of_day = dateutil.parser.parse(time_of_day).time()
                if from_time <= time_of_day <= to_time and (not camera_list or camera_id in camera_list):
                    selected_files.append("%s%s" % (prefix, filename))

    upload_json_to_s3(paths, ouput_file_name, selected_files)

    if return_files_flag:
        return selected_files


def format_time(timestr):
    return timestr.replace("-", ":")
