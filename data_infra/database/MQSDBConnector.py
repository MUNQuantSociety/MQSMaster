import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
import time
import logging

# Configure logging for better debugging and tracing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

class MQSDBConnector:
    """
    Thread-safe PostgreSQL connector class using connection pooling.
    This enhanced version includes connection health checks and more robust error handling.
    """

    def __init__(self):
        # Read environment variables for DB credentials
        self.db_host = os.getenv('host')
        self.db_port = int(os.getenv('port'))
        self.db_name = os.getenv('database')
        self.db_user = os.getenv('db_user')
        self.db_password = os.getenv('password')
        self.sslmode = os.getenv('sslmode', 'require')

        # Initialize the connection pool.
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=6,  # Adjust pool size as needed.
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                sslmode=self.sslmode
            )
            logging.info("Database connection pool created successfully.")
        except Exception as e:
            logging.error("Error creating connection pool: %s", e)
            raise e

        self.timeout = 600  # Example timeout (10 minutes)
        self.last_connection_time = time.time()

    def get_connection(self):
        """Retrieve a connection from the pool and verify it is active."""
        try:
            conn = self.pool.getconn()
            if conn.closed:
                logging.warning("Acquired a closed connection, retrying...")
                # self.pool.putconn(conn) # Return the closed conn before getting a new one
                conn = self.pool.getconn() # Get a new one
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return conn
        except Exception as e:
            logging.error("Error getting connection from pool: %s", e)
            return None

    def release_connection(self, conn):
        """Return the connection to the pool safely."""
        if conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                logging.error("Error releasing connection: %s", e)

    def close_all_connections(self):
        """Closes all connections in the pool."""
        try:
            self.pool.closeall()
            logging.info("All pooled connections closed successfully.")
        except Exception as e:
            logging.error("Error closing connections: %s", e)

    def execute_query(self, sql, values=None, fetch=False):
        """
        Executes a query with optional parameters.
        If fetch=True, returns results.
        """
        conn = self.get_connection()
        if not conn:
            return {'status': 'error', 'message': 'Could not obtain a database connection.'}

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql, values or ())
                if fetch:
                    result = cursor.fetchall()
                    conn.commit()
                    return {'status': 'success', 'message': 'Query executed successfully.', 'data': result}
                conn.commit()
                return {'status': 'success', 'message': 'Query executed successfully.'}
        except Exception as e:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error("Rollback failed: %s", rollback_error)
            logging.error("Error executing query: %s", e)
            return {'status': 'error', 'message': str(e)}
        finally:
            self.release_connection(conn)

    def inject_to_db(self, table, data, schema=None):
        """
        Inserts a single row into the specified table.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        schema_str = f"{schema}." if schema else ""
        sql = f"INSERT INTO {schema_str}{table} ({columns}) VALUES ({placeholders})"
        return self.execute_query(sql, tuple(data.values()))

    def bulk_inject_to_db(self, table, data: list[dict], conflict_columns: list[str] = None, schema=None):
        """
        Efficiently inserts multiple rows into a table using execute_values.
        Leverages 'ON CONFLICT DO NOTHING' if conflict_columns are provided.
        """
        if not data:
            return {'status': 'success', 'message': 'No data to insert.'}

        conn = self.get_connection()
        if not conn:
            return {'status': 'error', 'message': 'Could not obtain a database connection.'}

        try:
            with conn.cursor() as cursor:
                columns = data[0].keys()
                schema_str = f"{schema}." if schema else ""
                
                sql = f"INSERT INTO {schema_str}{table} ({', '.join(columns)}) VALUES %s"

                # Dynamically add the ON CONFLICT clause if conflict_columns are specified
                if conflict_columns:
                    sql += f" ON CONFLICT ({', '.join(conflict_columns)}) DO NOTHING"
                
                # Prepare data for execute_values
                values = [[row[col] for col in columns] for row in data]
                
                psycopg2.extras.execute_values(cursor, sql, values)
                inserted_count = cursor.rowcount
                conn.commit()
                
                return {'status': 'success', 'message': f'Successfully inserted or ignored {inserted_count} rows.'}

        except Exception as e:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error("Rollback failed: %s", rollback_error)
            logging.error("Error during bulk insert: %s", e)
            return {'status': 'error', 'message': str(e)}
        finally:
            self.release_connection(conn)


    def update_data(self, table, data, conditions=None, schema=None):
        """
        Updates records in a table based on provided conditions.
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for update.'}

        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"UPDATE {schema_str}{table} SET {set_clause} WHERE {where_clause}"
        values = list(data.values()) + list(conditions.values())
        return self.execute_query(sql, tuple(values))

    def delete_data(self, table, conditions=None, schema=None):
        """
        Deletes records matching the provided conditions.
        """
        if not conditions:
            return {'status': 'error', 'message': 'No conditions provided for deletion.'}

        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"DELETE FROM {schema_str}{table} WHERE {where_clause}"
        return self.execute_query(sql, tuple(conditions.values()))

    def read_db(self, table=None, columns='*', conditions=None, schema=None, sql=None):
        """
        Retrieves data from the database.
        If a custom SQL is provided, it will be executed directly.
        """
        if sql:
            return self.execute_query(sql, fetch=True)

        schema_str = f"{schema}." if schema else ""
        query = f"SELECT {columns} FROM {schema_str}{table}"
        values = None
        if conditions:
            where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
            query += f" WHERE {where_clause}"
            values = tuple(conditions.values())
        return self.execute_query(query, values, fetch=True)