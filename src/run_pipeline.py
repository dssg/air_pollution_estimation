
from src.d00_utils.data_retrieval import describe_s3_bucket, retrieve_single_video
from src.d00_utils.load_confs import load_parameters, load_paths

from src.d04_modelling.classify_objects import classify_objects

params = load_parameters()
paths = load_paths()

#describe_s3_bucket(paths)
video = retrieve_single_video('03675', '2019-06-20', '13:25:31.321785', paths, bool_keep_data=True)
