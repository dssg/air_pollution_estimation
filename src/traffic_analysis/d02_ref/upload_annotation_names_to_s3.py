import time as Time
from subprocess import Popen, PIPE

from traffic_analysis.d02_ref.ref_utils import upload_json_to_s3


def upload_annotation_names_to_s3(paths):
    """ Get the list of xml files from s3 and save a json on s3 containing the corresponding video filepaths
                    Args:
                        paths (dict): dictionary of paths from yml file

                    Returns:

        """

    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    prefix = "%s" % (paths['s3_annotations'])
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

    selected_files = []
    for file in files:
        if file:
            vals = file.split('_')
            if len(vals) == 4:
                vals = vals[1:]
            date = vals[0]
            time = vals[1].replace('-', ':')
            name = date + ' ' + time + '_' + vals[2]
            name = name.replace('.xml', '.mp4')
            selected_files.append(paths['s3_video'] + date + '/' + name)

    upload_json_to_s3(paths, 'annotations', selected_files)

    return
