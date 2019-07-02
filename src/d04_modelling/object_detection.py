from src.d00_utils.bbox_helpers import color_bboxes,bboxcvlib_to_bboxcv2
import numpy as np 
import cvlib

def detect_bboxes(frame:np.ndarray, model:str,confidence:float,
				  implementation:str=None,selected_labels:list=None) -> (list,list,list, list):
	'''Detect bounding boxes on a frame using specified model and optionally an implementation.
	bboxes returned in format (xmin, ymin, w, h). Colors are assigned to bboxes by the type. 

	Keyword arguments 

	frame -- one frame of a video 
	model -- specify the name of an object model to use 
	confidence -- only bboxes detected with above this confidence level will be returned 
	implementation -- specify the implementation of the model to use 
	selected_labels -- if a list of labels is supplied, only bboxes with these labels will be returned
	'''
	if implementation == 'cvlib':
		if model == 'yolov3_tiny':
			bboxes_cvlib, labels, confs = cvlib.detect_common_objects(frame, confidence=confidence,
												 model='yolov3_tiny')
			bboxes = [bboxcvlib_to_bboxcv2(bbox_cvlib) for bbox_cvlib in bboxes_cvlib]

	#sample for how other models/implementations could be added 
	elif implementation == 'some_other_impl':
		pass

	del_inds = []
	if selected_labels != None:
		for i, label in enumerate(labels): 
			#specify object types to ignore 
			if label not in selected_labels: del_inds.append(i)

		#delete items from lists
		for i in sorted(del_inds, reverse=True):
			del bboxes[i]; del labels[i]; del confs[i]
	#gen colors for remaining bboxes
	colors=color_bboxes(labels)

	return bboxes, labels, confs, colors
