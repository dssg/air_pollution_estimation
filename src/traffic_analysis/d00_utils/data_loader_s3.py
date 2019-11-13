import json
from traffic_analysis.d00_utils.load_confs import load_paths
import subprocess
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


class DataLoaderBlob:

    def __init__(self,
                 blob_credentials: dict):

        self.blob_credentials = blob_credentials
        self.client = BlobServiceClient.from_connection_string(blob_credentials['connection_string'])

        return

    def read_json(self, file_path):

        result = self.client.get_object(Bucket=self.bucket_name,
                                        Key=file_path)

        return json.loads(result['Body'].read().decode())

    def save_json(self, data, file_path):
        # TODO: ADD DOCUMENTATION. What type is data?

        blob_client = self.client.get_blob_client(container="pipeline", blob=file_path)
        try:
            blob_client.upload_blob(json.dumps(data))
        except:
            print("File already exists!")

    def file_exists(self, file_path):

        """
        try:
            with open(file_path, "rb") as data:
                self.client.upload_blob(data)

            return True

        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            else:
                raise ex
        """

    def download_file(self,
                      path_of_file_to_download,
                      path_to_download_file_to):

        self.client.download_file(Bucket=self.bucket_name,
                                  Key=path_of_file_to_download,
                                  Filename=path_to_download_file_to)

    def upload_file(self,
                    path_of_file_to_upload,
                    path_to_upload_file_to):

        try:
            blob_client = self.client.get_blob_client(container="pipeline", blob=path_to_upload_file_to)

            with open(path_of_file_to_upload, "rb") as data:
                blob_client.upload_blob(data)

        except:
            print("File already exists!")

        return

    def list_objects(self,
                     prefix=None) -> list:

        objects = self.client.list_objects_v2(Bucket=self.bucket_name,
                                              Prefix=prefix)

        return [file_dict['Key'] for file_dict in objects['Contents']]

    def move_file(self, old_path, new_path):
        paths = load_paths()
        s3_profile = paths['s3_profile']

        try:
            if old_path:
                old_filename = "s3://%s/%s" % (self.bucket_name, old_path)
                new_filename = "s3://%s/%s" % (self.bucket_name, new_path)
                res = subprocess.call(["aws", "s3", 'mv',
                                       old_filename,
                                       new_filename,
                                       '--profile',
                                       s3_profile])
        except Exception as e:
            print(e)
            return False
        return True

    def delete_folder(self, folder_path):
        paths = load_paths()
        s3_profile = paths['s3_profile']

        s3_path = "s3://%s/%s" % (self.bucket_name, folder_path)
        try:
            res = subprocess.call(["aws", "s3", 'rm',
                                   '--recursive',
                                   s3_path,
                                   '--profile',
                                   s3_profile])
        except Exception as e:
            print(e)
            return False
        return True
