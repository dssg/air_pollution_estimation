import os
import json

from traffic_analysis.d00_utils.data_retrieval import connect_to_bucket


def load_video_names_from_s3(ref_file, paths):
    """ Load the json file from ref on s3
            Args:
                ref_file (str): name of the reference file to be loaded
                paths (dict): dictionary of paths from yml file

            Returns:
                files (list): list of files to be downloaded from s3
        """

    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    # Download the json files and load the file names into a list
    local_path = os.path.join(paths["raw_video"], 'temp.json')
    my_bucket.download_file(paths['s3_ref'] + ref_file + '.json', local_path)
    with open(local_path, 'r') as f:
        files = json.load(f)

    # TODO: Handle case that annotations not available
    my_bucket.download_file(paths['s3_ref'] + 'annotations.json', local_path)
    with open(local_path, 'r') as f:
        files += json.load(f)

    os.remove(local_path)

    return files
