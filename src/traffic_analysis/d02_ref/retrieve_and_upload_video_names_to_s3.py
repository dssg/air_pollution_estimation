import datetime
import dateutil.parser

from traffic_analysis.d02_ref.ref_utils import generate_dates
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob


def retrieve_and_upload_video_names_to_s3(output_file_name: str,
                                          paths: dict,
                                          blob_credentials: dict,
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
    blob_video = paths['blob_video']
    to_date = dateutil.parser.parse(to_date).date()
    from_date = dateutil.parser.parse(from_date).date()
    from_time = dateutil.parser.parse(format_time(from_time)).time()
    to_time = dateutil.parser.parse(format_time(to_time)).time()
    selected_files = []

    dl = DataLoaderBlob(blob_credentials=blob_credentials)

    # Generate the list of dates
    dates = generate_dates(from_date, to_date)
    for date in dates:
        date = date.strftime('%Y%m%d')
        prefix = "%s%s-" % (blob_video, date)

        files, elapsed_time = dl.list_blobs(prefix=prefix)
        print('Extracting {} file names for date {} took {} seconds'.format(len(files),
                                                                            date,
                                                                            elapsed_time))

        if not files:
            continue

        for filename in files:
            if filename:
                res = filename.split('_')
                camera_id = res[-1][:-4]
                time_of_day = res[0].split('/')[-1]
                time_of_day = dateutil.parser.parse(time_of_day).time()
                if from_time <= time_of_day <= to_time and (not camera_list or camera_id in camera_list):
                    selected_files.append(filename)

    file_path = paths['blob_video_names'] + output_file_name + '.json'
    dl.save_json(data=selected_files, file_path=file_path)

    if return_files_flag:
        return selected_files



def format_time(timestr):
    return timestr.replace("-", ":")
