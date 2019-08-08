from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d02_ref.ref_utils import get_s3_video_path_from_xml_name


def upload_annotation_names_to_s3(paths,
                                  output_file_name: str,
                                  s3_credentials: dict):
    """ Get the list of xml files from s3 and save a json on s3 containing the corresponding video filepaths
                    Args:
                        paths (dict): dictionary of paths from yml file
                        s3_credentials:

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
        video_file = get_s3_video_path_from_xml_name(xml_file_name=file,
                                                     s3_creds=s3_credentials,
                                                     paths=paths)
        if(video_file):
            selected_files.append(video_file)

    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])
    file_path = paths['s3_video_names'] + output_file_name
    dl.save_json(data=selected_files, file_path=file_path)
