import numpy as np
import pandas as pd
import glob
import xml.etree.ElementTree as ElementTree

from src.d05_reporting.report_yolo import yolo_report_count_stats


def parse_annotations(paths, bool_print_summary=False):
    """ Parse the XML files containing the manual annotations
                Args:
                    xml_files: list of the XML files to parse
                    paths: dictionary of paths (needs 'annotations' path)
                    bool_print_summary: boolean for printing summaries of each parsed XML file
                Returns:
                    pandas dataframe containing the annotations
                Raises:
    """
    xml_files = glob.glob(paths['annotations'] + '*.xml')

    annotated_results = {'obj_id': [], 'frame_id': [], 'obj_bounds': [],
                         'obj_classification': [], 'parked': [], 'stopped': [],
                         'date': [], 'time': [], 'camera_id': [], 'video_id': []}

    for file_num, xml_file in enumerate(xml_files):
        root = ElementTree.parse(xml_file).getroot()

        name = xml_file.split('/')[-1]
        date = name.split('_')[1]
        time = name.split('_')[2].replace('-', ':')
        camera_id = name.split('_')[3][:-4]

        for track in root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):
                for frame in track.iter('box'):
                    annotated_results['obj_id'].append(int(track.attrib['id']))
                    annotated_results['frame_id'].append(int(frame.attrib['frame']))
                    annotated_results['obj_bounds'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                    for attr in frame.iter('attribute'):
                        if(attr.attrib['name'] == 'type'):
                            annotated_results['obj_classification'].append(attr.text)
                        else:
                            annotated_results[attr.attrib['name']].append(attr.text)

                    annotated_results['date'].append(date)
                    annotated_results['time'].append(time)
                    annotated_results['camera_id'].append(camera_id)
                    annotated_results['video_id'].append(file_num)

    df = pd.DataFrame.from_dict(annotated_results)

    if(bool_print_summary):
        print('Number of vehicles:')
        print(df.groupby('obj_classification')['obj_id'].nunique())
        print('Parked vehicles:')
        print(df.groupby('obj_id')['parked'].unique())
        stops_df = get_stop_counts(df)
        print('Number of Stops:')
        print(stops_df)

    return df


def get_stop_counts(annotations_df):
    """ Get the number of stops for each vehicle from the annotations dataframe
                    Args:
                        annotations_df: pandas dataframe containing the annotations
                    Returns:
                        pandas dataframe containing the vehicle ids and the number of stops
                    Raises:
    """
    ids, counts = [], []
    df_grouped = annotations_df.sort_values(['frame'], ascending=True).groupby('id')
    for group in df_grouped:
        bool_stopped = False
        num_stops = 0
        for val in group[1]['stopped'].tolist():
            if (val == 'true' and not bool_stopped):
                bool_stopped = True
            elif (val == 'false' and bool_stopped):
                num_stops += 1
                bool_stopped = False
        if (bool_stopped):
            num_stops += 1
        ids.append(group[1]['id'].tolist()[0])
        counts.append(num_stops)
    stops_df = pd.DataFrame(data=np.array(list(zip(ids, counts))), columns=['object_id', 'num_stops'])
    return stops_df


def report_true_count_stats(annotations_df):
    '''Report summary statistics for the output of YOLO on one video.

        Keyword arguments:
        yolo_df -- pandas df containing formatted output of YOLO for one video (takes the output of yolo_output_df())

        Returns:
        obj_counts_frame: counts of various types of objects per frame
        video_summary: summary statistics over whole video


        '''
    dfs = []
    grouped = annotations_df.groupby('video_id')

    for name, group in grouped:
        types = group.groupby('obj_id')['obj_classification'].unique()
        types = [t[0] for t in types]
        vals, counts = np.unique(types, return_counts=True)
        df = pd.DataFrame([counts], columns=vals)
        df['camera_id'] = group['camera_id'].values[0]
        df['date'] = group['date'].values[0]
        df['time'] = group['time'].values[0]
        df['video_id'] = group['video_id'].values[0]
        dfs.append(df)

    df = pd.concat(dfs)
    return df.fillna(0)


def get_count_accuracies(paths, annotations_df, yolo_df):
    print('Calculating Accuracies...')

    yolo_counts_df = yolo_report_count_stats(yolo_df)
    yolo_counts_df.sort_values("video_id", inplace=True)
    yolo_counts_df = yolo_counts_df[yolo_counts_df['metric'] == 'mean']
    true_counts_df = report_true_count_stats(annotations_df)
    true_counts_df.sort_values("video_id", inplace=True)

    df = pd.DataFrame()
    # Bus accuracy
    df['bus_diff'] = yolo_counts_df['bus'] - true_counts_df['bus']
    df['car_diff'] = yolo_counts_df['car'] - true_counts_df['car']
    df['truck_diff'] = yolo_counts_df['truck'] - true_counts_df['truck']
    df['motorbike_diff'] = yolo_counts_df['motorbike'] - true_counts_df['motorbike']

    return