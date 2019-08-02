import xml.etree.ElementTree as ET

from traffic_analysis.d00_utils.load_confs import load_paths, load_credentials
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

        self.data_loader_s3 = DataLoaderS3(s3_credentials=creds[paths['s3_creds']],
                                           bucket_name=paths['bucket_name'])

        return


    def get_train_and_test(self, train_fraction):

        for dataset in self.datasets:
            self.parse_mapping[dataset]()


        return

    def parse_detrac_data(self):

        print('Parsing detrac dataset...')

        xml_files = self.data_loader_s3.list_objects(prefix=self.paths['s3_detrac_annotations'])
        for xml_file in xml_files:
            self.parse_detrac_xml_file(xml_file)

        return

    def parse_detrac_xml_file(self, xml_file):

        path = self.paths['temp_annotation'] + xml_file.split('/')[-1]

        self.data_loader_s3.download_file(path_of_file_to_download=xml_file,
                                          path_to_download_file_to=path)

        tree = ET.parse(xml_file.split('/')[-1])
        img_name = path.split('/')[-1][:-4]

        height = tree.findtext("./size/height")
        width = tree.findtext("./size/width")

        objects = [img_name, width, height]
        print(objects)

        return

    def parse_cvat_data(self):

        print('Parsing cvat dataset...')

        return


paths = load_paths()
creds = load_credentials()

dl = DataLoader(datasets=[TransferDataset.detrac], creds=creds, paths=paths)
dl.get_train_and_test(.8)
