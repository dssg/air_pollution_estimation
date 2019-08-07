from traffic_analysis.d00_utils.create_sql_tables import create_sql_tables
from traffic_analysis.d00_utils.load_confs import load_paths

paths = load_paths()
create_sql_tables(db_vehicle_types_name=paths['db_vehicle_types'],
                  db_cameras_name=paths['db_cameras'],
                  db_frame_level_name=paths['db_frame_level'], 
                  db_video_level_name=paths['db_video_level'],
                  drop=True)
