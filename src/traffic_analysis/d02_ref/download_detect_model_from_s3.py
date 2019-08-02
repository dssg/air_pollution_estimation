import os
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


def download_detect_model_from_s3(model_name: str,
                                  paths: dict,
                                  s3_credentials: dict):
    """ Retrieves required files from s3 folder for detection model_name specified in params
        Args:
            model_name: dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            s3_credentials: s3 credentials
    """
    # only cary out if model_name has not been downloaded before
    local_folder_path_model = os.path.join(paths["local_detect_model"], model_name)

    if not os.path.exists(local_folder_path_model):
        os.makedirs(local_folder_path_model)

        dl = DataLoaderS3(s3_credentials,
                          bucket_name=paths['bucket_name'])"])

        files_to_download = dl.list_objects(prefix=paths['s3_detect_model'] + model_name)

        # download each file from s3 to local
        for path_of_file_to_download in files_to_download:
            s3_file_path, file_name = os.path.split(path_of_file_to_download)
            path_to_download_file_to = os.path.join(local_folder_path_model, file_name)
            dl.download_file(path_of_file_to_download=path_of_file_to_download,
                             path_to_download_file_to=path_to_download_file_to)
