import datetime

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def update_video_level_table(analyser,
                             frame_level_df=None,
                             file_names=None,
                             paths=None,
                             creds=None,
                             return_data=False):
    """ Update the video level table on the database based on the videos in the files list
                Args:
                    frame_level_df (dataframe): dataframe containing the frame level stats, if none then
                    this is loaded from the database using the file names
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    creds (dict): dictionary of credentials from yml file
                    return_data: For debugging it might be useful to return the video level df

                Returns:

    """
    db_obj = DataLoaderSQL(creds=creds, paths=paths)

    if frame_level_df is None:
        # Build the sql string
        filter_string = ''

        for filename in file_names:
            name = filename.split('/')[-1]
            camera_id, date_time = parse_video_or_annotation_name(name)
            filter_string += f"(camera_id='{camera_id}' AND video_upload_datetime='{str(date_time)}') OR "

        filter_string = filter_string[:-4]
        sql_string = "SELECT * FROM {} WHERE {};".format(
            paths['db_frame_level'], filter_string)
        frame_level_df = db_obj.select_from_table(sql=sql_string)

        bboxes = []
        for x, y, w, h in zip(frame_level_df['bbox_x'].values, frame_level_df['bbox_y'].values, frame_level_df['bbox_w'].values, frame_level_df['bbox_h'].values):
            bboxes.append([x, y, w, h])
        frame_level_df['bboxes'] = bboxes
        frame_level_df.drop('bbox_x', axis=1, inplace=True)
        frame_level_df.drop('bbox_y', axis=1, inplace=True)
        frame_level_df.drop('bbox_w', axis=1, inplace=True)
        frame_level_df.drop('bbox_h', axis=1, inplace=True)

    # Create video level table and add to database
    video_level_df = analyser.construct_video_level_df(frame_level_df)
    if video_level_df.empty:
        return
    video_level_df['creation_datetime'] = datetime.datetime.now()

    db_obj.add_to_sql(df=video_level_df, table_name=paths['db_video_level'])

    if return_data:
        return video_level_df
