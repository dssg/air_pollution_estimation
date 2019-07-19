import os
from src.traffic_analysis.d00_utils.get_project_directory import get_project_directory
from src.traffic_analysis.d00_utils.data_retrieval import connect_to_bucket


def retrieve_detect_model_from_s3(params, paths):
    """ Retrieves required files from s3 folder for detection model specified in params
        Args:
            params (dict): dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
    """

    model = params['yolo_model']
    local_filepath_model = os.path.join(paths['detect_model'], model)

    if not os.path.exists(local_filepath_model):  # download model files from s3 if local model filepath doesn't exist
        # make local file path
        os.makedirs(local_filepath_model)

        # get location of files in s3 bucket
        my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])
        s3_filepath_model = "ref/model_conf/" + model

        # get list of all files in the s3 folder
        objects = my_bucket.objects.filter(Prefix=s3_filepath_model)
        files = [obj.key for obj in objects]

        # download each file from s3 to local
        for filename in files:
            path, fn = os.path.split(filename)
            local_filepath_file = os.path.join(local_filepath_model, fn)
            my_bucket.download_file(filename, local_filepath_file)

    return
