from kedro.pipeline import Pipeline, node
from data_collection.d01_data.collect_tims_data import get_tims_data_and_upload_to_s3
from data_collection.d01_data.collect_video_data import collect_camera_videos
from data_collection.d01_data.collect_video_data import upload_videos

# TODO: Can we make this a data driven pipeline

data_collection_pipeline = Pipeline([
    node(
        func=get_tims_data_and_upload_to_s3,
        inputs=[],
        outputs=[],
        name='tims_upload'),
    # TODO: Change collect_camera_videos function to take in the data collection parameter instead of names variables
    node(
        func=collect_camera_videos,
        inputs=['data_collection'],
        outputs=[],
        name='tims_upload'),
    # TODO: Change upload_videos function to take in the data collection parameter instead of names variables
    node(
        func=upload_videos,
        inputs=['data_collection'],
        outputs=[],
        name='tims_upload')
    ])


# TODO: Once this is working, delete the main files in the data collection package.
# TODO: Should we move this pipeline to the data_collection package?
