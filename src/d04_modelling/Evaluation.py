import numpy as np
import xml.etree.ElementTree as ET

def evaluate_yolo_predictions(videos, yolo_output):

    for video in videos:
        # load the annotated xml file
        root = ET.parse('../../annotations/' + video + '.xml').getroot()

        annotated_results = {'id': [], 'type': [], 'x_coords': [], 'y_coords': [], 'width': [], 'height': [], 'parked': [], 'num_stops': [], 'stopped_frames': []}

        num_vehicles = 0
        num_parked = 0
        num_stops = 0
        types = []

        for track in root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):

                annotated_results['id'].append(int(track.attrib['id']))

                print(track.attrib)

                for frame_num, frame in enumerate(track.iter('box')):
                    print(frame.attrib)



                    for attr in frame.iter('attribute'):
                        print(attr.attrib)

                        if(attr.attrib['name'] == 'parked' and attr.text == 'true'):
                            b_parked = True

                        if (attr.attrib['name'] == 'type' and frame_num == 0):
                            types.append(attr.text)

                        if (attr.attrib['name'] == 'stopped' and attr.text == 'true'):
                            b_stopped = True

                        if(b_stopped and attr.attrib['name'] == 'stopped' and attr.text == 'false'):
                            num_stops += 1

                if(b_parked):
                    num_parked += 1

                if (b_stopped):
                    num_stops += 1

        print('Number of vehicles: ' + str(num_vehicles))
        print('Number of parked vehicles: ' + str(num_parked))
        print('Number of stops: ' + str(num_stops))
        vals, counts = np.unique(types, return_counts=True)
        print('Vehicle Types: ' + str(vals))
        print('Type Counts: ' + str(counts))

    return


evaluate_yolo_predictions(['video001'], None)