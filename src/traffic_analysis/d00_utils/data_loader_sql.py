import pandas as pd
import psycopg2
import io


class DataLoaderSQL():
    def __init__(self, creds, paths):

        db_host = paths['db_host']
        db_name = paths['db_name']
        db_user = creds['postgres']['user']
        db_pass = creds['postgres']['passphrase']

        self.connection_string = "password=%s user=%s dbname=%s host=%s" % (
        db_pass, db_user, db_name, db_host)

        self.conn = None
        self.cursor = None

    def open_connection(self):
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(self.connection_string)
            # self.conn.op
        self.cursor = self.conn.cursor()


    def add_to_sql(self, df, table_name):
        self.open_connection()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        # null values become ''
        self.cursor.copy_from(output, table_name, null="")
        self.conn.commit()


    def add_column_to_sql_table(self, table_name, new_column_name, data_type):
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

    def update_sql_table(self, table_name, values, condition):
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

    def execute_raw_sql_query(self, sql):
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
