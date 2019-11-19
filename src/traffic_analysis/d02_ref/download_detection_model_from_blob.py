import os
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob


def download_detection_model_from_blob(model_name: str,
                                       paths: dict,
                                       blob_credentials: dict):
    """ Retrieves required files from s3 folder for detection model_name specified in params
        Args:
            model_name: dictionary of parameters from yml file
            paths (dict): dictionary of paths from yml file
            blob_credentials: blob credentials
    """

    local_folder_path_model = os.path.join(
        paths['local_detection_model'], model_name)

    if not os.path.exists(local_folder_path_model):
        os.makedirs(local_folder_path_model)

    if not os.listdir(local_folder_path_model):
        dl = DataLoaderBlob(blob_credentials)

        files_to_download, elapsed_time = dl.list_blobs(
            prefix=paths['blob_detection_model'] + model_name)

        # download each file from blob to local
        for path_of_file_to_download in files_to_download:
            blob_file_path, file_name = os.path.split(path_of_file_to_download)
            path_to_download_file_to = os.path.join(
                local_folder_path_model, file_name)
            dl.download_blob(path_of_file_to_download=path_of_file_to_download,
                             path_to_download_file_to=path_to_download_file_to)
