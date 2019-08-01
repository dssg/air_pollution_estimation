import pandas as pd
import psycopg2
import io


class db():
    def __init__(self, conn_string):
        self.connection_string = conn_string
        self.conn = None
        self.cursor = None

    def open_connection(self):
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(self.connection_string)
            # self.conn.op
        self.cursor = self.conn.cursor()

    def create_tables(self, drop=False):
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
                video_upload_datetime timestamp,
                frame_id INTEGER,
                vehicle_id INTEGER,
                vehicle_type VARCHAR(255),
                confidence FLOAT,
                bbox_x FLOAT,
                bbox_y FLOAT,
                bbox_w FLOAT,
                bbox_h FLOAT
            )
            """,

            """
            CREATE TABLE video_stats (
                camera_id VARCHAR(255),
                video_upload_datetime timestamp,
                vehicle_type VARCHAR(255),
                counts FLOAT,  
                stops FLOAT,
                starts FLOAT
            )
            """
        ]
        try:
            self.open_connection()

            if(drop_commands is not None):
                for command in drop_commands:
                    self.cursor.execute(command)

            for command in commands:
                self.cursor.execute(command)

            self.conn.commit()
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            return False

        finally:
            self.cursor.close()
            if self.conn:
                self.conn.close()
        return True


    def save_data(self, df, table_name):
        self.open_connection()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        print(contents)
        # null values become ''
        self.cursor.copy_from(output, table_name, null="")
        self.conn.commit()

    def add_column(self, table_name, new_column_name, data_type):
        self.open_connection()

        results = None
        try:
            sql = "ALTER TABLE %s ADD COLUMN %s %s;" % (
                table_name, new_column_name, data_type)

            self.cursor.execute(sql)
            self.conn.commit()

            return True

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.conn.rollback()
        finally:
            self.cursor.close()
            if self.conn:
                self.conn.close()
        return False

    def update_table(self, table_name, values, condition):
        self.open_connection()

        try:
            sql = "UPDATE %s SET %s WHERE %s;" % (
                table_name, values[0], condition)
            result = self.cursor.execute(sql, (values[1]))
            print(result)
            self.conn.commit()

            return True
        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.conn.rollback()
        finally:
            self.cursor.close()
            if self.conn:
                self.conn.close()
        return False

    def execute_raw_query(self, sql):
        self.open_connection()

        results = None
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            col_names = list(map(lambda x: x[0], self.cursor.description))
            results = pd.DataFrame(results, columns=col_names)
            self.conn.commit()

        except(Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.conn.rollback()
        finally:
            self.cursor.close()
            if self.conn:

                self.conn.close()
        return results


if __name__ == "__main__":
    pass
