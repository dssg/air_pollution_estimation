
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3

from enum import Enum
class TransferDataset(Enum):
    detrac = 1
    cvat = 2


class DataLoader(object):

    def __init__(self, datasets, creds, paths):
        self.datasets = datasets
        self.creds = creds
        self.paths = paths
        self.parse_mapping = {TransferDataset.detrac: self.parse_detrac_data,
                              TransferDataset.cvat: self.parse_cvat_data}
        self.data_loader_s3 = DataLoaderS3(s3_credentials=, bucket_name=)

        return


    def get_train_and_test(self, train_fraction):

        for dataset in self.datasets:
            self.parse_mapping[dataset]()


        return

    def parse_detrac_data(self):

        print('Parsing detrac dataset...')

        return

    def parse_cvat_data(self):

        print('Parsing cvat dataset...')

        return



dl = DataLoader(datasets=[], creds=creds, paths=paths)
