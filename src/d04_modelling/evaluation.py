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
                    df: pandas dataframe containing the annotations
                Raises:
    """
    xml_files = glob.glob(paths['annotations'] + '*.xml')

    annotated_results = {'obj_id': [], 'frame_id': [], 'obj_bounds': [],
                         'obj_classification': [], 'parked': [], 'stopped': [],
                         'date': [], 'time': [], 'camera_id': [], 'video_id': []}

    for file_num, xml_file in enumerate(xml_files):
        root = ElementTree.parse(xml_file).getroot()

        name = xml_file.split('/')[-1]
        try:
            date = name.split("_")[1]
            time = name.split("_")[2].split('.')[0]
            camera_id = name.split('_')[3][:-4]
        except:
            date = name.split("_")[0]
            time = name.split("_")[1].split('.')[0]
            camera_id = name.split('_')[2][:-4]

        for track in root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):
                for frame in track.iter('box'):
                    annotated_results['obj_id'].append(int(track.attrib['id']))
                    annotated_results['frame_id'].append(int(frame.attrib['frame']))
                    annotated_results['obj_bounds'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                    for attr in frame.iter('attribute'):
                        # If name is 'type' then index the dictionary using 'obj_classification'
                        if(attr.attrib['name'] == 'type'):
                            annotated_results['obj_classification'].append(attr.text)
                        # Else just use the name for indexing
                        else:
                            annotated_results[attr.attrib['name']].append(attr.text)

                    annotated_results['date'].append(date)
                    annotated_results['time'].append(time)
                    annotated_results['camera_id'].append(camera_id)
                    annotated_results['video_id'].append(file_num)

    df = pd.DataFrame.from_dict(annotated_results)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
    df['time'] = pd.to_datetime(df['time'], format='%H-%M-%S').dt.time

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
    df_grouped = annotations_df.sort_values(['frame'], ascending=True).groupby('video_id')
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
    '''Report the true counts for multiple annotated videos.

        Keyword arguments:
        annotations_df -- pandas df containing the formatted output of the XML files
                          (takes the output of parse_annotations())

        Returns:
        df: dataframe containing the true counts for each video
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
        dfs.append(df)

    df = pd.concat(dfs)
    return df.fillna(0)


def report_count_differences(annotations_df, yolo_df, bool_plots=True):
    '''Report the difference in counts between yolo and the annotations for multiple videos.

            Keyword arguments:
            annotations_df -- pandas df containing the formatted output of the XML files
                              (takes the output of parse_annotations())
            yolo_df -- pandas df containing formatted output of YOLO for multiple videos
                       (takes the output of yolo_output_df())

            Returns:
            df: dataframe containing the difference in counts for each video
    '''
    yolo_counts_df = yolo_report_count_stats(yolo_df)
    yolo_counts_df.sort_values(["camera_id", "date", "time"], inplace=True)
    yolo_counts_df = yolo_counts_df[yolo_counts_df['metric'] == 'mean']
    true_counts_df = report_true_count_stats(annotations_df)
    true_counts_df.sort_values(["camera_id", "date", "time"], inplace=True)

    categories_to_compare = ['bus', 'car', 'truck', 'motorbike']
    diff_df = pd.DataFrame()

    for category in categories_to_compare:
        if(category not in yolo_counts_df.columns):
            yolo_counts_df[category] = 0
        if (category not in true_counts_df.columns):
            true_counts_df[category] = 0

        diff_df['p-y_' + category] = yolo_counts_df[category] - true_counts_df[category]

    assert (yolo_counts_df['camera_id'].values == true_counts_df['camera_id'].values).all(), \
        "camera IDs do not match in report_count_differences()"
    diff_df['camera_id'] = yolo_counts_df['camera_id']
    assert (yolo_counts_df['date'].values == true_counts_df['date'].values).all(), \
        "dates do not match in report_count_differences()"
    diff_df['date'] = yolo_counts_df['date']
    assert (yolo_counts_df['time'].values == true_counts_df['time'].values).all(), \
        "times do not match in report_count_differences()"
    diff_df['time'] = yolo_counts_df['time']

    if(bool_plots):
        pass

    return diff_df
