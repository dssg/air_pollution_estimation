import pandas as pd


def parse_annotation(xml_root) -> pd.DataFrame:
    """Returns annotation xml file as a pandas dataframe
    """

    annotated_result = {'vehicle_id': [],
                        'frame_id': [],
                        'bboxes': [],
                        'vehicle_type': [],
                        'parked': [],
                        'stopped': [],
                        'start_frame': 0,
                        'stop_frame': 0}

    for task in root.iter('task'):
        for task_info in task.iter():
            if task_info.tag == 'start_frame': 
                annotated_result['start_frame'] = task_info.text
            if task_info.tag == 'stop_frame':
                annotated_result['stop_frame'] = task_info.text
            

    for track in xml_root.iter('track'):
        if track.attrib['label'] == 'vehicle':
            for frame in track.iter('box'):
                annotated_result['vehicle_id'].append(int(track.attrib['id']))
                annotated_result['frame_id'].append(int(frame.attrib['frame']))
                annotated_result['bboxes'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                for attr in frame.iter('attribute'):
                    # If name is 'type' then index the dictionary using 'vehicle_type'
                    if attr.attrib['name'] == 'type':
                        annotated_result['vehicle_type'].append(attr.text)
                    # Else just use the name for indexing
                    else:
                        annotated_result[attr.attrib['name']].append(attr.text)
