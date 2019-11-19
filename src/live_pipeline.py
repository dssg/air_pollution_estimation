from datetime import datetime
from traffic_analysis.d07_pipelines.create_pipeline import create_pipeline
from traffic_analysis.d00_utils.load_confs import load_parameters

current_date = datetime.now().date()
output_file_name = f"{current_date}"
from_date = str(current_date)
to_date = str(current_date)
from_time = "00-00-00"
to_time = "23-59-59"
params = load_parameters()

create_pipeline(load_ref_file=True,
                output_file_name=output_file_name,
                from_date=from_date,
                to_date=to_date,
                from_time=from_time,
                to_time=to_time,
                chunk_size=params['chunk_size'],
                move_to_processed_folder=True,
                delete_processed_videos=True,
                camera_list=params["camera_list"]
                )
