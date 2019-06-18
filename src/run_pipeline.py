from src.d00_utils.data_retrieval import retrieve_videos_from_s3, describe_s3_bucket

describe_s3_bucket()
videos = retrieve_videos_from_s3(from_date='2019-06-06', to_date='2019-06-07')
