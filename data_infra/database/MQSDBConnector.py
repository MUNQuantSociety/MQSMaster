import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class MQSDBConnector:
    """
    Thread-safe PostgreSQL connector class.
    Uses connection pooling for efficiency.
    """

    def __init__(self):
        # Read environment variables for DB credentials
        self.db_host = os.getenv('host')
        self.db_port = int(os.getenv('port'))
        self.db_name = os.getenv('database')
        self.db_user = os.getenv('username')
        self.db_password = os.getenv('password')
        self.sslmode = os.getenv('sslmode', 'require')

        # Connection pooling: allows multiple threads to share connections safely
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, 
            maxconn=5,  # Adjust as needed
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=self.sslmode
        )

        self.timeout = 600  # 10 minute timeouts
        self.last_connection_time = time.time()

    def get_connection(self):
        """Retrieve a connection from the pool."""
        try:
            return self.pool.getconn()
        except Exception as e:
            print("Error getting connection:", e)
            return None

    def release_connection(self, conn):
        """Return connection to the pool."""
        if conn:
            self.pool.putconn(conn)

    def close_all_connections(self):
        """Closes all pooled connections."""
        self.pool.closeall()

    def execute_query(self, sql, values=None, fetch=False):
        """
        Executes a query with optional parameters.
        If fetch=True, returns results.
        """
        conn = self.get_connection()
        if not conn:
            return {'status': 'error', 'message': 'Could not get database connection'}

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql, values or ())
                if fetch:
                    result = cursor.fetchall()
                    return {'status': 'success', 'message': 'Query executed', 'data': result}
                conn.commit()
                return {'status': 'success', 'message': 'Query executed successfully'}
        except Exception as e:
            conn.rollback()
            return {'status': 'error', 'message': str(e)}
        finally:
            self.release_connection(conn)

    def inject_to_db(self, table, data, schema=None):
        """
        Inserts a row into the specified table.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        schema_str = f"{schema}." if schema else ""
        sql = f"INSERT INTO {schema_str}{table} ({columns}) VALUES ({placeholders})"
        return self.execute_query(sql, tuple(data.values()))

    def update_data(self, table, data, conditions=None, schema=None):
        """
        Updates records in a table based on given conditions.
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for update'}

        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"UPDATE {schema_str}{table} SET {set_clause} WHERE {where_clause}"

        values = list(data.values()) + list(conditions.values())
        return self.execute_query(sql, tuple(values))

    def delete_data(self, table, conditions=None, schema=None):
        """
        Deletes records matching conditions.
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for deletion'}

        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"DELETE FROM {schema_str}{table} WHERE {where_clause}"

        return self.execute_query(sql, tuple(conditions.values()))

    def read_db(self, table=None, columns='*', conditions=None, schema=None, sql=None):
        """
        Retrieves data from the database.
        """
        if sql:
            return self.execute_query(sql, fetch=True)

        schema_str = f"{schema}." if schema else ""
        query = f"SELECT {columns} FROM {schema_str}{table}"
        if conditions:
            where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
            query += f" WHERE {where_clause}"
            values = tuple(conditions.values())
        else:
            values = None

        return self.execute_query(query, values, fetch=True)
