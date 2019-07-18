from traffic_analysis.d02_ref.ref_utils import upload_json_to_s3
from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3


def upload_annotation_names_to_s3(paths):
    """ Get the list of xml files from s3 and save a json on s3 containing the corresponding video filepaths
                    Args:
                        paths (dict): dictionary of paths from yml file

                    Returns:

        """

    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    prefix = "%s" % (paths['s3_annotations'])

    # fetch video filenames
    elapsed_time, files = get_names_of_folder_content_from_s3(bucket_name, prefix, s3_profile)
    print('Extracting {} file names took {} seconds'.format(len(files),
                                                            elapsed_time))

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
