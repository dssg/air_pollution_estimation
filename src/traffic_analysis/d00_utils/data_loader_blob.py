import json
import time
import subprocess
from traffic_analysis.d00_utils.load_confs import load_paths
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


class DataLoaderBlob:

    def __init__(self,
                 blob_credentials: dict):

        self.blob_credentials = blob_credentials
        self.client = BlobServiceClient.from_connection_string(blob_credentials['connection_string'])

        return

    def read_json(self, file_path):

        blob_client = self.client.get_blob_client(container="pipeline", blob=file_path)
        try:
            result = blob_client.download_blob()
        except:
            print("Failed to download " + file_path)
            return

        return json.loads(result.readall())

    def save_json(self, data, file_path):

        blob_client = self.client.get_blob_client(container="pipeline", blob=file_path)
        try:
            blob_client.upload_blob(json.dumps(data))
        except:
            print("Replacing JSON file: " + str(file_path))
            self.delete_blobs([file_path])
            blob_client.upload_blob(json.dumps(data))


    def file_exists(self, file_path):


        blob = self.list_blobs(prefix=file_path)

        if(blob):
            return True
        else:
            return False


    def download_blob(self,
                      path_of_file_to_download,
                      path_to_download_file_to):


        try:
            blob_client = self.client.get_blob_client(container="pipeline", blob=path_of_file_to_download)

            with open(path_to_download_file_to, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

        except:
            print("Could not download " + path_of_file_to_download)

        return

    def upload_blob(self,
                    path_of_file_to_upload,
                    path_to_upload_file_to):

        try:
            blob_client = self.client.get_blob_client(container="pipeline", blob=path_to_upload_file_to)

            with open(path_of_file_to_upload, "rb") as data:
                blob_client.upload_blob(data)

        except:
            print("File already exists!")

        return

    def list_blobs(self,
                   prefix=None):

        start = time.time()
        container_client = self.client.get_container_client(container="pipeline")
        blobs = container_client.list_blobs(name_starts_with=prefix)
        blob_names = []
        for blob in blobs:
            blob_names.append(blob.name)

        elapsed_time = time.time() - start

        return blob_names, elapsed_time

    def copy_blob(self, file_to_move, new_file, paths):

        path_to_download_file_to = paths["temp_video"] + \
                                   file_to_move.split('/')[-1]

        self.download_blob(path_of_file_to_download=file_to_move, path_to_download_file_to=path_to_download_file_to)
        self.upload_blob(path_of_file_to_upload=path_to_download_file_to, path_to_upload_file_to=new_file)

        return True

    def delete_blobs(self, blobs):

        for blob in blobs:

            try:
                blob_client = self.client.get_blob_client(container="pipeline", blob=blob)
                blob_client.delete_blob()

            except:
                print("Could not delete " + str(blob))
                return False

        return True
