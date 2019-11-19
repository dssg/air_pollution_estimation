from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob


def load_video_names_from_blob(ref_file: str,
                               paths: dict,
                               blob_credentials: dict) -> list:
    """ Load the json file from ref on blob
        Args:
            ref_file: name of the reference file to be loaded
            paths: dictionary of paths from yml file
            blob_credentials: blob credentials

        Returns:
            files: list of files to be downloaded from blob
    """

    dl = DataLoaderBlob(blob_credentials=blob_credentials)

    files = dl.read_json(file_path=paths['blob_video_names'] + ref_file + '.json')

    # avoid duplication
    files = list(set(files))

    return files
