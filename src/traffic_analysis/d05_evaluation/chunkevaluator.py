from src.traffic_analysis.d05_evaluation.singleevaluator import FrameLevelEvaluator, VideoLevelEvaluator

#TODO: change refs to d05_reporting to d05_evaluation
import numpy as np 
import pandas as pd
import glob
#create evaluator class(xml files, video level table, frame level table)
#subclasses frame level and video level
#evaluate_all functiion will evaluate frame level and video level by default 
#to do this it will call frame_level_evaluation and video_level_evaluation funct

class ChunkEvaluator(): 
	def __init__(self, annotation_xml_paths:list, frame_level_dfs:list, video_level_dfs:list, params:dict): 
		assert len(annotation_xmls) == len(frame_level_dfs) == len(video_level_dfs)

		self.num_videos = len(annotation_xmls)
		self.annotation_xml_paths = annotation_xmls
		self.frame_level_dfs = frame_level_dfs
		self.video_level_df = video_level_dfs
		self.params = params

		self.frame_level_diffs 
		self.video_level_diff_df

	def evaluate(self, frame_level = True, 
				video_level = True, 
				video_eval_stat_types = ['mse','mean']
				): 

		for i,xml_path in enumerate(self.annotation_xml_paths): 
		    xml_name = re.split(r"\\|/",xml_path)[-1]
	        xml_root = ElementTree.parse(xml_path).getroot()

			if frame_level:
				frame_level_evaluator = FrameLevelEvaluator(xml_root, xml_name,
													  		self.frame_level_dfs[i], 
													  		params) 

			if video_level: 
				video_level_evaluator = VideoLevelEvaluator(xml_root,xml_name,
															self.video_level_dfs[i],
															params)
				self.video_level_diff_df = video_level_evaluator.evaluate_video()

				stats_dfs = []
				for stat_type in video_eval_stat_types: 
					stat_fcn = getattr(self,stat_type)
					stat_dfs.append(stat_fcn(self.video_level_diff_df,self.params['video_level_stats']))
				stats_df = pd.concat(stat_dfs, axis = 0)
				print("stats df: ", stats_df)

	def mse(self, df:pd.DataFrame, stat_cols:list): 
		"""
		"""
		#call compute_mean
		#create df: cols with ['n_videos', 'mse']
		n_rows = len(df)

		mse_df = pd.DataFrame()
		mse_df["n"] = n_rows

		for stat in stat_cols: 
			mse["mse"+stat] = (df["y_pred-y_"+stat]**2)/n_rows
		return mse

	def mean(self): 
		#cols with ['n_videos', 'mse']
		pass


	def plot_evaluation_stats(self): 
		pass


if __name__ == '__main__':
	