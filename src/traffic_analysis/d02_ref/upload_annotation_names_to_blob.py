from traffic_analysis.d02_ref.ref_utils import get_names_of_folder_content_from_s3
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob
from traffic_analysis.d02_ref.ref_utils import get_s3_video_path_from_xml_name


def upload_annotation_names_to_blob(paths,
                                  output_file_name: str,
                                  blob_credentials: dict,
                                  verbose=True) -> dict:
    """ Get the list of xml files from s3 and save a json on s3 containing the corresponding video filepaths
    Args:
        paths: dictionary of paths from yml file
        s3_credentials: dictionary of credentials from yml file
    """

    prefix = "%s" % (paths['blob_annotations'])

    dl_blob = DataLoaderBlob(blob_credentials=blob_credentials)

    # fetch annotation filenames
    annotation_files, elapsed_time = dl_blob.list_blobs(prefix)
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

    file_path = paths['s3_video_names'] + output_file_name + '.json'
    dl_blob.save_json(data=selected_files, file_path=file_path)
