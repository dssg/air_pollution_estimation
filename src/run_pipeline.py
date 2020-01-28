from traffic_analysis.d07_pipelines.create_pipeline import create_pipeline
from traffic_analysis.d00_utils.load_confs import load_parameters

params = load_parameters()
output_file_name = params['ref_file_name']

create_pipeline(output_file_name=output_file_name,
                from_date=params['from_date'], to_date=params['to_date'],
                from_time=params['from_time'], to_time=params['to_time'],
                chunk_size=params['chunk_size'],
                camera_list=params["camera_list"],
                construct_ref_file=params['load_ref_file'])
