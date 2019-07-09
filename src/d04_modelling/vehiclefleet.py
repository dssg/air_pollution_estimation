import numpy as np
import collections
import matplotlib.pyplot as plt
from src.d00_utils.bbox_helpers import color_bboxes, bbox_intersection_over_union,vectorized_intersection_over_union, bboxcv2_to_bboxcvlib
from src.d00_utils.stats_helpers import time_series_smoother

class VehicleFleet():
    def __init__(self,bboxes:np.ndarray,labels:np.ndarray,confs:np.ndarray):
        """
        dims: (7,4)
        bboxes in cv2 format
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
        label_confs = [label +', id=' + str(i) + ', '+ str(format(self.confs[i] * 100, '.2f')) + '%' for i,label in enumerate(self.labels)]
        return label_confs


    def compute_colors(self) -> list:
        return color_bboxes(self.labels)


    def compute_counts(self):
        return collections.Counter(self.labels)


    def compute_iou_video(self, interval: int = 1):
        #iterate along time axis; consider 2 timepoints t0,t1 at a time 
        #compare bboxes at t0,t1; output the iou
        self.iou_interval = interval
        num_frames = self.bboxes.shape[2]
        num_vehicles = self.bboxes.shape[0]
        iou_video = np.zeros((num_vehicles,num_frames - interval))
        #compare iou for in between frames 
        # for i in range(0,num_frames-interval, interval):
        for i in range(0,num_frames-interval):

            bboxes_time_t0 = self.bboxes[:,:,i]
            bboxes_time_t1 = self.bboxes[:,:,i+interval]

            for j in range(num_vehicles): 

                iou_video[j,i] = bbox_intersection_over_union(bboxcv2_to_bboxcvlib(bboxes_time_t0[j]),
                                                       bboxcv2_to_bboxcvlib(bboxes_time_t1[j]))

            # iou_video[:,i] = vectorized_intersection_over_union(bboxcv2_to_bboxcvlib(bboxes_time_t0, vectorized = True),
                                                              # bboxcv2_to_bboxcvlib(bboxes_time_t1, vectorized = True))
            # assert iou_video_ind == iou_video[0,i], str(iou_video_ind) + str(iou_video[0,i])
        self.iou_video = iou_video
        # print(self.bboxes[4,:,:])
        # print(iou_video[4,:])
        return 


    def smooth_iou_video(self, smoother_method:str, **smoother_settings): 
        # num_vehicles = self.iou_video.shape[0]

        self.smoothed_iou_video = time_series_smoother(self.iou_video, 
                                                    method = smoother_method, 
                                                    **smoother_settings)
        return 


    def plot_iou_video(self, fig_path, smoothed = False, vehicle_ids = None):
        """Visualize the iou_video
        """
        iou = self.smoothed_iou_video if smoothed else self.iou_video

        num_vehicles,num_ious = iou.shape[0], iou.shape[1]

        if vehicle_ids is None:
            vehicles_ids = range(num_vehicles) 

        #plot each vehicle
        vehicle_colors = np.array(self.compute_colors()) / 255
        iou_inds = np.arange(num_ious)
        for i in range(num_vehicles):
            iou_vehicle = iou[i,:]
            mask1,mask2 = np.isnan(iou_vehicle),np.isfinite(iou_vehicle)
            plt.plot(iou_inds[~mask1 & mask2], iou_vehicle[~mask1 & mask2], 
                    # color = vehicle_colors[i], 
                    label = "vehicle " + str(i) + "; type " + self.labels[i])
        plt.legend(loc = 'lower right')
        plt.xlabel("IOU over all frames in video, interval = " + str(self.iou_interval))
        plt.savefig(fig_path)
        plt.close()





