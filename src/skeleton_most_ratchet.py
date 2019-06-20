import yaml
from d00_utils.download_s3_cam_data import s3_to_local_mp4
from d04_modelling.classify_objects import classify_objects

os.chdir(".")
with open('../../conf/base/paths.yml') as f:
    path = yaml.safe_load(f)['s3_paths']

with open('../../conf/base/parameters.yml') as f:
    params = yaml.safe_load(f)['modelling']


if __name__ == "__main__":
    # take stored video data on cloud and put on local directory
    local_vid_path = s3_to_local_mp4(camera, date, time, local_vid_dir, extension='mp4')

    # detect objects with cvlib
    obj_bounds, obj_labels, obj_label_confidences = classify_objects(local_vid_path, vid_time_length=10, make_video=True)

    # reformat output cvlib np array as a dataframe
    ## INSERT CAROLINES CODE ##

    # take statistical information from dataframe
    ## INSERT CAROLINES CODE ##

    # upload video data and stats to api
    ## INSERT FOOMZ CODE ##