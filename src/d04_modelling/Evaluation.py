import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

paths = {'annotations': '../../annotations/'}

def evaluate_yolo_predictions(videos, yolo_output, paths):

    for video in videos:
        # load the annotated xml file
        root = ET.parse(paths['annotations'] + video + '.xml').getroot()

        annotated_results = {'id': [], 'frame': [], 'occluded': [], 'bounds': [],
                             'type': [], 'parked': [], 'stopped': []}

        num_vehicles = 0
        num_parked = 0
        num_stops = 0
        types = []

        for track in root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):
                for frame in track.iter('box'):
                    annotated_results['id'].append(int(track.attrib['id']))
                    annotated_results['frame'].append(int(frame.attrib['frame']))
                    annotated_results['occluded'].append(int(frame.attrib['occluded']))
                    annotated_results['bounds'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                    for attr in frame.iter('attribute'):
                        annotated_results[attr.attrib['name']].append(attr.text)

        df = pd.DataFrame.from_dict(annotated_results)

        print('Number of vehicles:')
        print(df.groupby('type')['id'].nunique())

        print('Parked vehicles:')
        print(df.groupby('id')['parked'].unique())

        print('Number of stops: ' + str(num_stops))
        vals, counts = np.unique(types, return_counts=True)
        print('Vehicle Types: ' + str(vals))
        print('Type Counts: ' + str(counts))

    return


evaluate_yolo_predictions(['video001'], None, paths)