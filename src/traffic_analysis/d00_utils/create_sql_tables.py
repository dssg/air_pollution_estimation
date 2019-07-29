import psycopg2


def create_tables(conn_string, drop=False):
    connection_string = conn_string
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()

    drop_commands = None
    if(drop):
        drop_commands = [
            "DROP TABLE vehicle_types, cameras, frame_stats, video_stats CASCADE;"
        ]

    commands = [
        """
        CREATE TABLE vehicle_types(
            id SERIAL,
            vehicle_type VARCHAR(255),
            vehicle_type_id INTEGER PRIMARY KEY
        )
        """,
        """
        CREATE TABLE cameras(
            id SERIAL,
            latitude FLOAT, 
            longitude FLOAT,
            borough INTEGER,
            tfl_camera_id INTEGER,
            camera_name VARCHAR(500)
        )
        """,
        """
        CREATE TABLE frame_stats (
            camera_id VARCHAR(255),
            frame_id INTEGER,
            datetime timestamp,
            obj_classification VARCHAR(255),
            confidence FLOAT,
            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER
        )
        """,

        """
        CREATE TABLE video_stats (
            counts FLOAT,
            vehicle_type VARCHAR(255),
            camera_id VARCHAR(255),
            datetime timestamp,
            starts FLOAT,
            stops FLOAT
        )
        """
    ]
    try:

        if(drop_commands is not None):
            for command in drop_commands:
                cursor.execute(command)

        for command in commands:
            cursor.execute(command)

        conn.commit()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
        return False

    finally:
        cursor.close()
        if conn:
            conn.close()
    return True
