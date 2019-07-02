from data_collection.d01_data.collect_video_data import collect_camera_videos
import os

if __name__ == "__main__":
    # local data folder
    setup_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', '..')
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')
    cam_file = os.path.join(setup_dir, 'data', '00_ref', 'cam_file.json')
    collect_camera_videos(
        local_video_dir=video_dir,
        cam_file=cam_file)
