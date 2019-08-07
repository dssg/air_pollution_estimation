import boto3
import json
import xml.etree.ElementTree as ElementTree
from botocore.exceptions import ClientError


class DataLoaderS3:

    def __init__(self,
                 s3_credentials: dict,
                 bucket_name: str):

        self.s3_credentials = s3_credentials
        self.bucket_name = bucket_name

        client = boto3.client('s3',
                              aws_access_key_id=s3_credentials['aws_access_key_id'],
                              aws_secret_access_key=s3_credentials['aws_secret_access_key']
                              )
        self.client = client

    def read_json(self, file_path):
        
        result = self.client.get_object(Bucket=self.bucket_name,
                                        Key=file_path)

        return json.loads(result['Body'].read().decode())

    def save_json(self, data, file_path):

        self.client.put_object(Body=(bytes(json.dumps(data).encode('UTF-8'))),
                               Bucket=self.bucket_name,
                               Key=file_path)

    def read_xml(self, file_path):
        result = self.client.get_object(Bucket=self.bucket_name,
                                        Key=file_path)
        xml_as_str = result['Body'].read().decode()
        xml_root = ElementTree.fromstring(xml_as_str)
        return xml_root


    def file_exists(self, file_path):

        try:
            self.client.get_object(Bucket=self.bucket_name,
                                   Key=file_path)
            return True

        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return False
            else:
                raise ex

    def download_file(self,
                      path_of_file_to_download,
                      path_to_download_file_to):

        self.client.download_file(Bucket=self.bucket_name,
                                  Key=path_of_file_to_download,
                                  Filename=path_to_download_file_to)

    def list_objects(self,
                     prefix=None)->list:

        objects = self.client.list_objects_v2(Bucket=self.bucket_name,
                                              Prefix=prefix)

        return [file_dict['Key'] for file_dict in objects['Contents']]
