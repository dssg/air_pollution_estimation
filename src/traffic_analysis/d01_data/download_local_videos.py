import datetime
import dateutil.parser

from traffic_analysis.d02_ref.ref_utils import generate_dates
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob

from traffic_analysis.d00_utils.load_confs import (load_credentials,
                                                   load_parameters, load_paths)



def format_time(timestr):
    return timestr.replace("-", ":")

params = load_parameters()
paths = load_paths()
creds = load_credentials()
blob_credentials = creds[paths['blob_creds']]

from_date = '2019-12-03'
to_date = '2019-12-03'

from_time = '00-00-00'
to_time = '23-59-59'

camera_list = []

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

for filename in selected_files:
    path_to_download_file_to = paths["local_video"] + \
                               filename.split('/')[-1].replace(':', '-').replace(" ", "_")
    dl.download_blob(path_of_file_to_download=filename,
                     path_to_download_file_to=path_to_download_file_to)
