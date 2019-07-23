from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d00_utils.load_confs import load_parameters
import numpy as np 
import pandas as pd
import xml.etree.ElementTree as ElementTree
import re 
import collections
import datetime

class SingleEvaluator(): 
    def __init__(self, xml_root, xml_name, params): 
        """
        """
        self.xml_root = xml_root
        self.camera_id, self.video_upload_datetime = parse_video_or_annotation_name(xml_name)
        self.annotated_result = {'vehicle_id': [],
                                'frame_id': [], 
                                'bboxes': [],
                                'vehicle_type': [], 
                                'parked': [], 
                                'stopped': []}
        self.selected_labels = params['selected_labels']


    def parse_annotation(self) -> pd.DataFrame: 
        """
        Returns annotated xml file as a pandas dataframe
        """
        for track in self.xml_root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):
                for frame in track.iter('box'):
                    self.annotated_result['vehicle_id'].append(int(track.attrib['id']))
                    self.annotated_result['frame_id'].append(int(frame.attrib['frame']))
                    self.annotated_result['bboxes'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                    for attr in frame.iter('attribute'):
                        # If name is 'type' then index the dictionary using 'vehicle_type'
                        if(attr.attrib['name'] == 'type'):
                            self.annotated_result['vehicle_type'].append(attr.text)
                        # Else just use the name for indexing
                        else:
                            self.annotated_result[attr.attrib['name']].append(attr.text)

        ground_truth_df = pd.DataFrame.from_dict(self.annotated_result)
        ground_truth_df['video_upload_datetime'] = self.video_upload_datetime
        ground_truth_df['camera_id'] = self.camera_id
        return ground_truth_df


class FrameLevelEvaluator(SingleEvaluator): 
    """
    """
    def __init__(self, xml_root, xml_name, frame_level_df:pd.DataFrame): 
        super().__init__(xml_root, xml_name)
        self.ground_truth_df = super(FrameLevelEvaluator, self).parse_annotations()

    def evaluate_video(self): 
        pass

    def plot_video(self): 
        pass

####################################
class VideoLevelEvaluator(SingleEvaluator):
    def __init__(self, xml_root,xml_name, video_level_df:pd.DataFrame, params):
        super().__init__(xml_root, xml_name, params)
        print(self.xml_root)
        self.ground_truth_df = super().parse_annotation()
        self.video_level_df = video_level_df
        self.video_level_column_order = params["video_level_column_order"]

    def fill_and_sort_by_vehicle_types(self, df): 
        """
        """
        for column_name in self.video_level_column_order: 
            if column_name not in df.columns: 
                df[column_name] = 0
        df = df[self.video_level_column_order]

        # insert missing vehicle types as row
        current_vehicle_types = df['vehicle_type'].values
        for vehicle_type in self.selected_labels:
            # need to append rows 
            if vehicle_type not in current_vehicle_types:
                #append type as new column
                new_row_dict = {(column_name):(df[column_name].iloc[0] if i<3 else 0) \
                                for i,column_name in enumerate(self.video_level_column_order)}
                new_row_dict['vehicle_type'] = vehicle_type
                df.loc[len(df) + 1] = new_row_dict
        df = df.sort_values(by='vehicle_type').reset_index()
        return df

    def evaluate_video(self): 
        """
        """
        true_stats_df = self.compute_true_video_level_stats()
        true_stats_df = self.fill_and_sort_by_vehicle_types(true_stats_df)
        self.video_level_df = self.fill_and_sort_by_vehicle_types(self.video_level_df)

        diff_columns = [col if i < 3 else 'y_pred-y_'+col  \
                        for i,col in enumerate(self.video_level_column_order)]
        diff_df = pd.DataFrame(columns = diff_columns)

        for stat in self.video_level_column_order[3:]:
            diff_df['y_pred-y_' + stat] = self.video_level_df[stat] - true_stats_df[stat]

        # print("camera id video df: ", self.video_level_df['camera_id'].iloc[0], 
            # "\n", "true_stats_df: ", true_stats_df['camera_id'].iloc[0])

        assert (self.video_level_df['camera_id'].iloc[0] == true_stats_df['camera_id'].iloc[0]), \
            "camera IDs do not match in VideoLevelEvaluator.evaluate_video()"
        diff_df['camera_id'] = self.video_level_df['camera_id']

        assert (self.video_level_df['video_upload_datetime'].iloc[0] == true_stats_df['video_upload_datetime'].iloc[0]), \
            "dates do not match in VideoLevelEvaluator.evaluate_video()"
        diff_df['video_upload_datetime'] = self.video_level_df['video_upload_datetime']

        diff_df['vehicle_type'] = self.video_level_df['vehicle_type']
        return diff_df

    def compute_true_video_level_stats(self): 
        """
        """
        counts_df = self.get_true_vehicle_counts()
        parked_df = self.get_true_parked_counts()
        stops_starts_df = self.get_true_stop_start_counts()

        counts_df['camera_id'] = self.camera_id
        counts_df['video_upload_datetime'] =  self.video_upload_datetime
        counts_df = counts_df.merge(parked_df, how='outer',
                                 on = "vehicle_type")
        counts_df = counts_df.merge(stops_starts_df, how='outer',
                                 on = "vehicle_type").fillna(0)

        counts_df = counts_df[self.video_level_column_order]
        return counts_df

    def get_true_vehicle_counts(self):
        '''Report the true counts for one annotated videos.

            Keyword arguments:
            annotations_df -- pandas df containing the formatted output of the XML files
                              (takes the output of parse_annotations())

            Returns:
            df: dataframe containing the true counts for each video
        '''
        types = self.ground_truth_df.groupby('vehicle_id')['vehicle_type'].unique()
        types = [t[0] for t in types]

        vals, counts = np.unique(types, return_counts=True)
        counts_df = pd.DataFrame(counts, index=vals)

        counts_df.index.name = 'vehicle_type'
        counts_df.reset_index(inplace=True)

        counts_df = counts_df.rename({0:"counts"},axis = "columns").fillna(0)

        return counts_df

    def get_true_stop_start_counts(self) -> (pd.DataFrame, pd.DataFrame):
        """ Get the number of stops for each vehicle from the annotations dataframe
        Args:
        Returns:
        Raises:
        """
        stop_counts, start_counts= [], []
        vehicle_dfs = self.ground_truth_df.sort_values(['frame_id'], ascending=True).groupby('vehicle_id')
        
        #compute number of starts stops for each vehicle in the video 
        for vehicle_id, vehicle_df in vehicle_dfs:
            vehicle_type = vehicle_df["vehicle_type"].tolist()[0]
            bool_stopped_prev = False

            for stopped_label in vehicle_df['stopped'].tolist():
                # convert bool string to boolean type 
                bool_stopped_current = True if stopped_label == 'true' else False

                if bool_stopped_current != bool_stopped_prev:
                    # going from moving to stopped
                    if bool_stopped_current:
                        stop_counts.append(vehicle_type)
                    # going from stopped to moving 
                    elif not bool_stopped_current: 
                        start_counts.append(vehicle_type)

                    bool_stopped_prev = bool_stopped_current

        # organize into df 
        stops_dict, starts_dict = collections.Counter(stop_counts), collections.Counter(start_counts)            
        stops_df = pd.DataFrame.from_dict(stops_dict,
                                          orient='index', columns=['stops'])
        starts_df = pd.DataFrame.from_dict(starts_dict,
                                          orient='index', columns=['starts'])

        stops_df.index.name = 'vehicle_type'
        stops_df.reset_index(inplace=True)

        starts_df.index.name = 'vehicle_type'
        starts_df.reset_index(inplace=True)


        return stops_df.merge(starts_df, how='outer',
                                 on = "vehicle_type")

    def get_true_parked_counts(self): 
        """
        """
        parked_counter = []
        for vehicle_id, vehicle_df in self.ground_truth_df.groupby('vehicle_id'): 
            parked_status = vehicle_df['parked'].tolist()[0]
            # convert str to bool 
            parked_boolean =  False if parked_status == 'false' else True 
            vehicle_type = vehicle_df['vehicle_type'].tolist()[0]

            if parked_boolean: 
                parked_counter.append(vehicle_type)

        parked_dict = collections.Counter(parked_counter)            
        parked_df = pd.DataFrame.from_dict(parked_dict,
                                          orient='index', columns=['parked'])
        parked_df.index.name = 'vehicle_type'
        parked_df.reset_index(inplace=True)

        return parked_df
