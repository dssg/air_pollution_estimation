import boto3
import json
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
