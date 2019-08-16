from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d02_ref.ref_utils import get_s3_video_path_from_xml_name


def upload_annotation_names_to_s3(paths,
                                  output_file_name: str,
                                  s3_credentials: dict,
                                  verbose=True) -> dict:
    """ Get the list of xml files from s3 and save a json on s3 containing the corresponding video filepaths
            Args:
                paths (dict): dictionary of paths from yml file
                s3_credentials:

            Returns: dictionary mapping the name of the annotation file to the name of the video file
    """

    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    prefix = "%s" % (paths['s3_cvat_annotations'])

    # fetch annotation filenames
    elapsed_time, annotation_files = get_names_of_folder_content_from_s3(bucket_name, prefix, s3_profile)
    if verbose: 
        print('Extracting {} file names took {} seconds'.format(len(annotation_files),
                                                            elapsed_time))
    selected_files = []
    for annotation_file in annotation_files:
        if annotation_file:
            stripped_annotation_file = annotation_file.replace(".xml", "")
            video_file = get_s3_video_path_from_xml_name(xml_file_name=stripped_annotation_file,
                                                        s3_creds=s3_credentials,
                                                        paths=paths)

            if(video_file):
                selected_files.append(video_file)

#    print("in annotations function, tried to get selected files")
#    print(selected_files)
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])
    file_path = paths['s3_video_names'] + output_file_name + ".json"
#    print("Saving json to", file_path)
    dl.save_json(data=selected_files, file_path=file_path)
