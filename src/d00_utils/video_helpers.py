import imageio
import numpy as np
import os

def write_mp4(local_mp4_dir:str, mp4_name:str,video:np.ndarray,fps:int):
	"""Write provided video to provided path

	Keyword arguments 

	local_mp4_dir -- path to directory to store vids in 
	mp4_name -- desired name for video. Please include .mp4 extension 
	fps -- provide the frames per second of the video 
	"""
	local_mp4_path_out = os.path.join(local_mp4_dir, mp4_name)
	imageio.mimwrite(local_mp4_path_out, video, fps=fps)
