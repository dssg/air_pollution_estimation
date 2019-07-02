#
# # # Explore Tracking Methods in OpenCV
# # Source: https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
#
#
# from __future__ import print_function
#
# from os import path
# import cv2
# from random import randint
#
# basepath=path.dirname(__file__) #path of current script
#
#
# def create_tracker_by_name(tracker_type):
#     # Create a tracker based on tracker name
#     if tracker_type == tracker_types[0]:
#     tracker = cv2.TrackerBoosting_create()
#     elif tracker_type == tracker_types[1]:
#     tracker = cv2.TrackerMIL_create()
#     elif tracker_type == tracker_types[2]:
#     tracker = cv2.TrackerKCF_create()
#     elif tracker_type == tracker_types[3]:
#     tracker = cv2.TrackerTLD_create()
#     elif tracker_type == tracker_types[4]:
#     tracker = cv2.TrackerMedianFlow_create()
#     elif tracker_type == tracker_types[5]:
#     tracker = cv2.TrackerGOTURN_create()
#     elif tracker_type == tracker_types[6]:
#     tracker = cv2.TrackerMOSSE_create()
#     elif tracker_type == tracker_types[7]:
#     tracker = cv2.TrackerCSRT_create()
#     else:
#     tracker = None
#     print('Incorrect tracker name')
#     print('Available trackers are:')
#     for t in tracker_types:
#       print(t)
#
#     return tracker
#
#
# def draw_bounding_boxes():
#     return None
# # In[18]:
#
#
# #load video frames
#
# if __name__ == '__main__':
#     tracker_types = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
#     video_path = path.abspath(path.join(basepath,"..", "data/sample_video_data/testvid.mp4"))
#
#     # Create a video capture object to read videos
#     cap = cv2.VideoCapture(video_path)
#
#     # Read first frame
#     success, frame = cap.read()
#
#     if not success:
#     print('Failed to read video')
#
#
#     ## Select boxes
#     bboxes = []
#     colors = []
#
#     # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
#     # So we will call this function in a loop till we are done selecting all objects
#     while True:
#         # draw bounding boxes over objects
#         # selectROI's default behaviour is to draw box starting from the center
#         # when fromCenter is set to false, you can draw box starting from top left corner
#         bbox = cv2.selectROI('MultiTracker', frame)
#         bboxes.append(bbox)
#         colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#         print("Press q to quit selecting boxes and start tracking")
#         print("Press any other key to select next object")
#         k = cv2.waitKey(0) & 0xFF
#         if (k == 113):  # q is pressed
#           break
#
#         print('Selected bounding boxes {}'.format(bboxes))
#
#
#     ### try CSRT
#
#     # Specify the tracker type
#     trackerType = "CSRT"
#
#     # Create MultiTracker object
#     multiTracker = cv2.MultiTracker_create()
#
#     # Initialize MultiTracker
#     for bbox in bboxes:
#     multiTracker.add(createTrackerByName(trackerType), frame, bbox)
#
#
#
#
#     # Process video and track objects
#     while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     # get updated location of objects in subsequent frames
#     success, boxes = multiTracker.update(frame)
#
#     # draw tracked objects
#     for i, newbox in enumerate(boxes):
#         p1 = (int(newbox[0]), int(newbox[1]))
#         p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
#         cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
#
#     # show frame
#         cv2.imshow('MultiTracker', frame)
#
#
#     # quit on ESC button
#     if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
#         break
#
#
# # ## Boosting
#
# # ## KCF
#
# # ## GOTURN
