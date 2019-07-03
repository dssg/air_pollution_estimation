import numpy as np

class VehicleFleet():
	def __init__(self,bboxes:np.ndarray,labels:np.ndarray,confs:np.ndarray):
		"""
		dims: (7,4)
		"""
		self.bboxes = np.expand_dims(bboxes,axis=2) #ADD A TIME DIMENSION
		self.labels = labels
		self.confs = confs


	def add_vehicles(self,new_bboxes:np.ndarray,new_labels:np.ndarray,new_confs:np.ndarray):
		"""
		job: append more vehicles, come up with na arrays for prev history
		"""
		current_time_t = self.bboxes.shape[2]
		num_new_vehicles = new_bboxes.shape[0]
		#create bboxes of all zeros to denote that the vehicle didn't exist at previous times
		new_vehicle_history = np.zeros((num_new_vehicles,4,current_time_t-1)) 
		new_bboxes = np.concatenate((np.expand_dims(new_bboxes,axis=2),new_vehicle_history),axis=2)

		self.bboxes = np.concatenate((self.bboxes,new_bboxes), axis=0)
		self.labels = np.concatenate((self.labels,new_labels), axis=0)
		self.confs = np.concatenate((self.confs,new_confs), axis=0)
		return


	def update_vehicles(self,bboxes_time_t):
		"""
		job: append in the time axis for all existing vehicles 
		"""
		self.bboxes = np.concatenate((self.bboxes,np.expand_dims(bboxes_time_t,axis=2)), axis=2)
		return


	def compute_label_confs(self): 
		label_confs = [label +' ' + str(format(self.confs[i] * 100, '.2f')) + '%' for i,label in enumerate(self.labels)]
		return label_confs

	def compute_colors(self):
		return color_bboxes(self.labels)

	def compute_stop_start(self):
		return


class Vehicle():
	def __init__(self,vehicle_id,bbox,label,conf):
		self.type = label
		self.detection_confidence = conf 
		self.unique_id = vehicle_id
		self.frame_appeared = 0 
		self.fraem_disappeared = 100


