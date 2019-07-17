from src.traffic_analysis.d05_evaluation.singleevaluator import FrameLevelEvaluator, VideoLevelEvaluator

#TODO: change refs to d05_reporting to d05_evaluation
import numpy as np 
import pandas as pd
import glob
#create evaluator class(xml files, video level table, frame level table)
#subclasses frame level and video level
#evaluate_all functiion will evaluate frame level and video level by default 
#to do this it will call frame_level_evaluation and video_level_evaluation funct

xml_files = glob.glob(paths['annotations'] + '*.xml')

class ChunkEvaluator(): 
	def __init__(self, annotation_xml_paths:list, frame_level_dfs:list, video_level_dfs:list): 
		assert len(annotation_xmls) == len(frame_level_dfs) == len(video_level_dfs)

		self.num_videos = len(annotation_xmls)
		self.annotation_xml_paths = annotation_xmls
		self.frame_level_dfs = frame_level_dfs
		self.video_level_df = video_level_dfs

	def evaluate(self, frame_level = True, video_level = True): 

		for i,xml_path in enumerate(self.annotation_xml_paths): 
		    xml_name = xml_path.split('/')[-1]
	        xml_root = ElementTree.parse(xml_path).getroot()

			if frame_level:
				frame_level_evaluator = FrameLevelEvaluator(xml_root, xml_name,
													  		self.frame_level_dfs[i]) 

			if video_level: 
				video_level_evaluator = VideoLevelEvaluator(xml_root,xml_name,
															self.video_level_dfs[i])

	def plot_evaluation_stats(self): 
		pass


if __name__ == '__main__':
	main()