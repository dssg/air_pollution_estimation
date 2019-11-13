import datetime
import dateutil.parser

from traffic_analysis.d02_ref.ref_utils import generate_dates
from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderBlob


def retrieve_and_upload_video_names_to_s3(output_file_name: str,
                                          paths: dict,
                                          s3_credentials: dict,
                                          from_date: str = '2019-06-01',
                                          to_date: str = str(
                                              datetime.datetime.now().date()),
                                          from_time: str = '00-00-00',
                                          to_time: str = '23-59-59',
                                          camera_list: list = None,
                                          return_files_flag=False):
    """Upload a json to s3 containing the filepaths for videos between the dates, times and cameras specified.

    Args:
        output_file_name: name of the json to be saved
        paths: dictionary containing temp_video, raw_video, s3_profile and bucket_name paths
        s3_credentials: S3 Credentials
        from_date: start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
        to_date: end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
        from_time: start time for retrieving videos, if None then will retrieve from the start of the day
        to_time: end time for retrieving videos, if None then will retrieve up to the end of the day
        camera_list: list of cameras to retrieve from, if None then retrieve from all cameras
    Returns:
        selected_files: if return_files_flag is True, will return list of S3 video paths 

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

    dl = DataLoaderBlob(s3_credentials,
                        bucket_name=paths['bucket_name'])
    file_path = paths['s3_video_names'] + output_file_name + '.json'
    dl.save_json(data=selected_files, file_path=file_path)

    if return_files_flag:
        return selected_files


def format_time(timestr):
    return timestr.replace("-", ":")
