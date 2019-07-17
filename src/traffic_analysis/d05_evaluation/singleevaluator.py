from src.traffic_analysis.d00_utils.video_helpers import parse_video_name
import numpy as np 
import pandas as pd
import xml.etree.ElementTree as ElementTree

class SingleEvaluator(): 
	def __init__(self, xml_root, xml_name, res_df): 
		self.xml_root = xml_root
		self.camera_id, self.video_upload_datetime = parse_video_name(xml_name)
		self.res_df = res_df
		self.annotated_result = {'obj_id': [], 
								'frame_id': [], 
								'obj_bounds': [],
                         		'obj_classification': [], 
                         		'parked': [], 
                         		'stopped': [],
                         		'video_upload_datetime': []}


	def parse_annotation(self) -> pd.DataFrame: 
		"""
		Returns annotated xml file as a pandas dataframe
		"""
		for track in self.xml_root.iter('track'):
            if(track.attrib['label'] == 'vehicle'):
                for frame in track.iter('box'):
                    annotated_result['obj_id'].append(int(track.attrib['id']))
                    annotated_result['frame_id'].append(int(frame.attrib['frame']))
                    annotated_result['obj_bounds'].append([float(frame.attrib['xtl']), float(frame.attrib['ytl']),
                                                        float(frame.attrib['xbr']), float(frame.attrib['ybr'])])
                    for attr in frame.iter('attribute'):
                        # If name is 'type' then index the dictionary using 'obj_classification'
                        if(attr.attrib['name'] == 'type'):
                            annotated_result['obj_classification'].append(attr.text)
                        # Else just use the name for indexing
                        else:
                            annotated_result[attr.attrib['name']].append(attr.text)

        ground_truth_df = pd.DataFrame.from_dict(annotated_result)
    	ground_truth_df['video_upload_datetime'] = self.video_upload_datetime
    	ground_truth_df['camera_id'] = self.camera_id
	    # df['time'] = pd.to_datetime(df['time'], format='%H-%M-%S').dt.time
	    return ground_truth_df


class FrameLevelEvaluator(SingleEvaluator): 
	def __init__(self, xml_root, frame_level_df:pd.DataFrame): 
		super().__init__(xml_root)
		self.ground_truth_df = self.parse_annotations()

	def evaluate_video(self): 
		return

	def plot_video(self): 
		pass


class VideoLevelEvaluator(SingleEvaluator):
	def __init__(self, xml_root, video_level_df:pd.DataFrame):
		super().__init__(xml_root, video_level_df)
		self.ground_truth_df = self.parse_annotations()

	def evaluate_video(self): 
		pass 

	def plot_video(self):
		pass 

	def get_true_stop_counts(self):
	    """ Get the number of stops for each vehicle from the annotations dataframe
	    Args:
	        annotations_df: pandas dataframe containing the annotations
	    Returns:
	        pandas dataframe containing the vehicle ids and the number of stops
	    Raises:
	    """
	    ids, counts = [], []
	    df_grouped = self.ground_truth_df.sort_values(['frame'], ascending=True).groupby('video_id')
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


	def get_true_vehicle_counts(self):
	    '''Report the true counts for multiple annotated videos.

	        Keyword arguments:
	        annotations_df -- pandas df containing the formatted output of the XML files
	                          (takes the output of parse_annotations())

	        Returns:
	        df: dataframe containing the true counts for each video
	    '''
	    dfs = []
	    grouped = self.ground_truth_df.groupby('video_id')

	    for name, group in grouped:
	        types = group.groupby('obj_id')['obj_classification'].unique()
	        types = [t[0] for t in types]
	        vals, counts = np.unique(types, return_counts=True)
	        df = pd.DataFrame([counts], columns=vals)
	        df['camera_id'] = group['camera_id'].values[0]
	        df['date'] = group['date'].values[0]
	        df['time'] = group['time'].values[0]
	        dfs.append(df)

	    df = pd.concat(dfs, sort=True)
	    return df.fillna(0)
