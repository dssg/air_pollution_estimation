import os
import sys
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from enum import Enum
import cv2

from traffic_analysis.d00_utils.load_confs import load_paths, load_credentials
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir, mp4_to_npy
from traffic_analysis.d02_ref.ref_utils import get_s3_video_path_from_xml_name
from traffic_analysis.d00_utils.get_project_directory import get_project_directory
from traffic_analysis.d04_modelling.transfer_learning.tensorflow_detection_utils import read_class_names


class TransferDataset(Enum):
    detrac = 1
    cvat = 2


class DataLoader(object):

    def __init__(self, datasets, creds, paths):
        self.datasets = datasets
        self.creds = creds
        self.paths = paths
        self.load_mapping = {TransferDataset.detrac: self.load_detrac_data,
                             TransferDataset.cvat: self.load_cvat_data}

        self.data_loader_s3 = DataLoaderS3(s3_credentials=creds[paths['s3_creds']],
                                           bucket_name=paths['bucket_name'])

        return


    def get_train_and_test(self, train_fraction):

        x, y = self.load_data_from_s3()

        x_train = x[:int(len(x) * train_fraction)]
        y_train = y[:int(len(x) * train_fraction)]

        x_test = x[int(len(x) * train_fraction):]
        y_test = y[int(len(x) * train_fraction):]

        return x_train, y_train, x_test, y_test

    def load_data_from_s3(self):

        self.clear_temp_folders()

        xs = []
        ys = []
        for dataset in self.datasets:
            x, y = self.load_mapping[dataset]()
            assert len(x) == len(y), "Mismatch in number of input and output pairs! " \
                                     "(Dataset: " + dataset.name + ")"
            xs += x
            ys += y

        # self.clear_temp_folders()

        return xs, ys

    def clear_temp_folders(self):
        delete_and_recreate_dir(self.paths['temp_annotation'])
        delete_and_recreate_dir(self.paths['temp_raw_images'])
        delete_and_recreate_dir(self.paths['temp_raw_video'])

    def load_detrac_data(self):

        print('Parsing detrac xmls...')
        y = []
        xml_files = self.data_loader_s3.list_objects(prefix=self.paths['s3_detrac_annotations'])
        count = 0
        for xml_file in xml_files:
            result = self.parse_detrac_xml_file(xml_file)
            if (result):
                y += result

            if count == 10:
                break
            count += 1

        print('Loading detrac images...')
        x = []
        for labels in y:
            image_num = labels.split(' ')[0].zfill(5)
            impath = labels.split(' ')[1]
            folder = impath.split('/')[-1][:9]
            print('---- downloading images from %s ----' % folder)

            file_to_download = self.paths['s3_detrac_images'] + \
                               folder + '/' + \
                               'img' + image_num + '.jpg'

            download_file_to = self.paths['temp_raw_images'] + \
                               folder + '_' + \
                               image_num + '.jpg'

            self.data_loader_s3.download_file(
                path_of_file_to_download=file_to_download,
                path_to_download_file_to=download_file_to)

            img = Image.open(download_file_to)
            img.load()
            x.append(np.asarray(img, dtype="int32"))

        return x, y

    def parse_detrac_xml_file(self, xml_file):

        project_dir = get_project_directory()
        image_dir = os.path.join(project_dir, self.paths['temp_raw_images'])

        xml_file_name = xml_file.split('/')[-1]
        xml_path = self.paths['temp_annotation'] + xml_file.split('/')[-1]

        class_names_path = os.path.join(self.paths['local_detection_model'], 'yolov3', 'coco.names')
        classes = read_class_names(class_names_path)

        try:
            self.data_loader_s3.download_file(path_of_file_to_download=xml_file,
                                              path_to_download_file_to=xml_path)
        except:
            print("Could not download file " + xml_file)

        root = ET.parse(xml_path).getroot()

        results = []
        # [image_index
        # image_path
        # image_width
        # image_height
        # label_index,
        # x_min,
        # y_min,
        # x_max,
        # y_max]

        im_width = 960
        im_height = 540

        for track in root.iter('frame'):
            frame_str = str(track.attrib['num']).zfill(5)
            im_path = os.path.join(image_dir, xml_file_name[:-4] + '_' + frame_str + '.jpg')
            result = str(track.attrib['num']) + \
                     ' ' + str(im_path) + \
                     ' ' + str(im_width) + \
                     ' ' + str(im_height)

            for frame_obj in track.iter('target'):
                vehicle_type = frame_obj.find('attribute').attrib['vehicle_type']
                if vehicle_type == 'van':
                    vehicle_type_idx = 2  # say vans are cars because we don't distinguish
                else:
                    for tick in range(len(classes)):
                        if classes[tick] == vehicle_type:
                            vehicle_type_idx = tick

                left = float(frame_obj.find('box').attrib['left'])
                top = float(frame_obj.find('box').attrib['top'])
                width = float(frame_obj.find('box').attrib['width'])
                height = float(frame_obj.find('box').attrib['height'])

                x_min = left
                y_min = top
                x_max = left + width
                y_max = top + height

                result += ' ' + str(vehicle_type_idx) + \
                          ' ' + str(x_min) + \
                          ' ' + str(y_min) + \
                          ' ' + str(x_max) + \
                          ' ' + str(y_max)

            results.append(result)

        if len(results) > 1:
            return results
        else:
            return None

    def load_cvat_data(self):

        print('Parsing cvat xmls...')
        y = []
        xml_files = self.data_loader_s3.list_objects(prefix=self.paths['s3_cvat_training_annotations'])
        vid_names = []
        for xml_file in xml_files:
            result, vid_name = self.parse_cvat_xml_file(xml_file)
            if(result):
                y += result
                vid_names.append(vid_name)

        print(len(y))
        print(vid_names)

        print('Loading cvat videos...')

        # Build a list of the videos needed
        video_set = set()
        image_path_in_y = []
        for labels in y:
            video_set.add(labels.split(' ')[1].split('/')[-1][:-10])
            image_path_in_y.append(labels.split(' ')[1])

        x = []
        for id in video_set:
            video, video_path = self.get_cvat_video(id)
            video_name = video_path.split('/')[-1][:-4]
            vidcap = cv2.VideoCapture(video_path)

            count = 0
            while vidcap.isOpened():
                success, image = vidcap.read()
                if success:
                    image_num = str(count).zfill(5)
                    im_path = os.path.join(self.paths['temp_raw_images'], video_name + '_' + image_num + '.jpg')
                    if im_path in image_path_in_y:
                        x.append(np.asarray(image, dtype="int32"))
                        print(np.asarray(image).shape)
                        cv2.imwrite(im_path, image)
                    count += 1
                else:
                    break
            vidcap.release()
        return x, y

    def get_cvat_video(self, xml_file_name):

        video_path = get_s3_video_path_from_xml_name(xml_file_name=xml_file_name,
                                                     s3_creds=self.creds[self.paths['s3_creds']],
                                                     paths=self.paths)
        if(video_path):
            download_file_to = os.path.join(self.paths['temp_raw_video'], xml_file_name + '.mp4')
            self.data_loader_s3.download_file(path_of_file_to_download=video_path,
                                              path_to_download_file_to=download_file_to)
            return mp4_to_npy(download_file_to), download_file_to
        else:
            return

    def parse_cvat_xml_file(self, xml_file):

        path = self.paths['temp_annotation'] + xml_file.split('/')[-1]
        print(path)

        try:
            self.data_loader_s3.download_file(path_of_file_to_download=xml_file,
                                              path_to_download_file_to=path)
        except:
            print("Could not download file " + xml_file)

        root = ET.parse(path).getroot()
        vid_name = path.split('/')[-1][:-4]
        im_dir = self.paths['temp_raw_images']

        im_width = 352
        im_height = 288

        class_names_path = os.path.join(self.paths['local_detection_model'], 'yolov3', 'coco.names')
        classes = read_class_names(class_names_path)

        frame_dict = {}

        for track in root.iter('track'):
            for frame in track.iter('box'):
                frame_num = frame.attrib['frame']

                if(frame_num not in frame_dict):
                    frame_name = str(frame_num).zfill(5)
                    im_path = os.path.join(im_dir, vid_name + '_' + frame_name + '.jpg')
                    frame_dict[frame_num] = str(frame_num) + ' ' + \
                                            str(im_path) + ' ' + \
                                            str(im_width) + ' ' + \
                                            str(im_height)
                if track.attrib['label'] == 'vehicle':
                    vehicle_type = frame.findall('attribute')[2].text
                    if vehicle_type == 'van':
                        vehicle_type_idx = 2  # say vans are cars because we don't distinguish
                    else:
                        for tick in range(len(classes)):
                            if classes[tick] == vehicle_type:
                                vehicle_type_idx = tick

                    x_min = float(frame.attrib['xtl'])
                    y_min = float(frame.attrib['ytl'])
                    x_max = float(frame.attrib['xbr'])
                    y_max = float(frame.attrib['ybr'])

                    frame_dict[frame_num] += ' ' + str(vehicle_type_idx) + \
                                  ' ' + str(x_min) + \
                                  ' ' + str(y_min) + \
                                  ' ' + str(x_max) + \
                                  ' ' + str(y_max)

        results = []
        for key in frame_dict:
            results.append(frame_dict[key])

        if len(results) > 1:
            return results, vid_name
        else:
            return None
