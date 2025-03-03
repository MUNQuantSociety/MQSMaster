"""
MQSDBConnector.py
-----------------
A robust PostgreSQL connector class for your trading bot.
Loads DB credentials from the root .env.
"""

import os
import time
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load from the project's root .env (contains DB credentials + MEMBER_AUTH_TOKEN)
load_dotenv()

class MQSDBConnector:
    """
    A robust DB connector class for PostgreSQL.
    Provides methods for connect, read, inject (insert), update, delete,
    and ensures connections are properly managed.
    """

    def __init__(self):
        # Read environment variables for DB credentials from the root .env
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', 5432))
        self.db_name = os.getenv('DB_NAME', 'market_db')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')

        # Internal placeholders
        self.connection = None
        self.cursor = None

        # Time-based connection logic
        self.last_connection_time = None
        self.timeout = 300  # 5 minutes

    def connect(self):
        """
        Establish a new connection if not already connected or if timed out.
        """
        if self.connection is not None:
            try:
                # Ping the connection. If it's alive and not timed out, return.
                if (time.time() - self.last_connection_time) < self.timeout:
                    return {'status': 'success', 'message': 'Already connected.'}
                else:
                    self.close_connection()
            except:
                self.close_connection()

        try:
            self.connection = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.last_connection_time = time.time()
            return {'status': 'success', 'message': 'Connected to PostgreSQL successfully.'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def check_connection(self):
        """
        Checks if the current connection is still valid and not timed out.
        Reconnects if necessary.
        """
        if self.connection is None:
            return self.connect()

        try:
            self.cursor.execute("SELECT 1;")
            if (time.time() - self.last_connection_time) >= self.timeout:
                self.close_connection()
                return self.connect()
            self.last_connection_time = time.time()
            return {'status': 'success', 'message': 'Connection is valid.'}
        except:
            self.close_connection()
            return self.connect()

    def inject_to_db(self, table, data, schema=None):
        """
        Inserts a row into the given table. 'data' is a dict of {column: value}.
        """
        check_res = self.check_connection()
        if check_res['status'] == 'error':
            return check_res

        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            schema_str = f"{schema}." if schema else ""
            sql = f"INSERT INTO {schema_str}{table} ({columns}) VALUES ({placeholders})"

            self.cursor.execute(sql, tuple(data.values()))
            self.connection.commit()
            return {'status': 'success', 'message': f"Data inserted into {table} successfully."}
        except Exception as e:
            self.connection.rollback()
            return {'status': 'error', 'message': str(e)}

    def delete_data(self, table, conditions=None, schema=None):
        """
        Deletes rows that match 'conditions' dict from a table.
        E.g., conditions = {'id': 123}
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for deletion.'}

        check_res = self.check_connection()
        if check_res['status'] == 'error':
            return check_res

        try:
            schema_str = f"{schema}." if schema else ""
            where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
            sql = f"DELETE FROM {schema_str}{table} WHERE {where_clause}"

            self.cursor.execute(sql, tuple(conditions.values()))
            self.connection.commit()
            return {'status': 'success', 'message': f"Data deleted from {table} successfully."}
        except Exception as e:
            self.connection.rollback()
            return {'status': 'error', 'message': str(e)}

    def update_data(self, table, data, conditions=None, schema=None):
        """
        Updates rows matching 'conditions' dict with columns in 'data' dict.
        E.g., data = {'price': 150.0}, conditions = {'ticker': 'AAPL'}
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for update.'}

        check_res = self.check_connection()
        if check_res['status'] == 'error':
            return check_res

        try:
            schema_str = f"{schema}." if schema else ""
            set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
            where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
            sql = f"UPDATE {schema_str}{table} SET {set_clause} WHERE {where_clause}"

            values = list(data.values()) + list(conditions.values())
            self.cursor.execute(sql, tuple(values))
            self.connection.commit()
            return {'status': 'success', 'message': f"Data updated in {table} successfully."}
        except Exception as e:
            self.connection.rollback()
            return {'status': 'error', 'message': str(e)}

    def read_db(self, table=None, columns='*', conditions=None, schema=None, sql=None):
        """
        Retrieves rows from the database.
        If 'sql' is given, it overrides table/columns/conditions usage.
        Otherwise, builds a SELECT query.
        """
        check_res = self.check_connection()
        if check_res['status'] == 'error':
            return {'status': 'error', 'message': check_res['message'], 'data': None}

        try:
            if sql:
                self.cursor.execute(sql)
            else:
                schema_str = f"{schema}." if schema else ""
                query = f"SELECT {columns} FROM {schema_str}{table}"
                if conditions:
                    # conditions can be a list of strings, e.g. ["ticker='AAPL'","price>100"]
                    where_clause = " AND ".join(conditions)
                    query += f" WHERE {where_clause}"
                self.cursor.execute(query)

            rows = self.cursor.fetchall()
            return {
                'status': 'success',
                'message': 'Data fetched successfully.',
                'data': rows
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'data': None
            }

    def close_connection(self):
        """
        Closes the connection to the PostgreSQL database if open.
        """
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
