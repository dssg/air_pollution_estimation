from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


def load_video_names_from_s3(ref_file,
                             paths,
                             s3_credentials: dict):
    """ Load the json file from ref on s3. If annotations for those
            Args:
                ref_file (str): name of the reference file to be loaded
                paths (dict): dictionary of paths from yml file
                s3_credentials: s3 credentials

            Returns:
                files (list): list of files to be downloaded from s3
        """

    dl = DataLoaderS3(s3_credentials=s3_credentials,
                      bucket_name=paths['bucket_name'])

    files = dl.read_json(file_path=paths['s3_video_names'] + ref_file + '.json')

    annotations_path = paths['s3_video_names'] + 'annotations.json'
    if dl.file_exists(annotations_path):
        annotation_video_names = dl.read_json(file_path=paths['s3_video_names'] + 'annotations.json')
        files += annotation_video_names

    # avoid duplication
    files = list(set(files))

    return files
