import datetime
import time as Time
from subprocess import Popen, PIPE

from traffic_analysis.d02_ref.ref_utils import upload_json_to_s3
from traffic_analysis.d02_ref.ref_utils import generate_dates


def retrieve_and_upload_video_names_to_s3(save_name,
                                          paths,
                                          from_date='2019-06-01',
                                          to_date=str(datetime.datetime.now().date()),
                                          from_time='00-00-00',
                                          to_time='23-59-59',
                                          camera_list=None):
    """Upload a json to s3 containing the filepaths for videos between the dates, times and cameras specified.

        Args:
            save_name (str): name of the json to be saved
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
    from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d').date()
    to_date = datetime.datetime.strptime(to_date, '%Y-%m-%d').date()
    from_time = datetime.datetime.strptime(from_time, '%H-%M-%S').time()
    to_time = datetime.datetime.strptime(to_time, '%H-%M-%S').time()
    selected_files = []

    # Generate the list of dates
    dates = generate_dates(from_date, to_date)
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        prefix = "%s%s/" % (s3_video, date)
        start = Time.time()

        # fetch video filenames
        ls = Popen(["aws", "s3", 'ls', 's3://%s/%s' % (bucket_name, prefix),
                    '--profile',
                    s3_profile], stdout=PIPE)
        p1 = Popen(['awk', '{$1=$2=$3=""; print $0}'],
                   stdin=ls.stdout, stdout=PIPE)
        p2 = Popen(['sed', 's/^[ \t]*//'], stdin=p1.stdout, stdout=PIPE)
        ls.stdout.close()
        p1.stdout.close()
        output = p2.communicate()[0]
        p2.stdout.close()
        files = output.decode("utf-8").split("\n")
        end = Time.time()
        print(end - start, len(files))
        if not files:
            break
        for filename in files:
            if filename:
                res = filename.split('_')
                camera_id = res[-1][:-4]
                time_of_day = res[0].split(".")[0]
                time_of_day = datetime.datetime.strptime(
                    time_of_day, '%Y-%m-%d %H:%M:%S').time()
                if from_time <= time_of_day <= to_time and (not camera_list or camera_id in camera_list):
                    selected_files.append("%s%s" % (prefix, filename))

    upload_json_to_s3(paths, save_name, selected_files)

    return

