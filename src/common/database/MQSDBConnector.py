import logging
import os
import time
from typing import Any

import psycopg2
from psycopg2.extensions import connection
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

# Configure logging for better debugging and tracing.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Load environment variables
if (load_dotenv() is False):
    logging.error("Failed to load environment variables")


class MQSDBConnector:
    """
    Thread-safe PostgreSQL connector class using connection pooling.
    This enhanced version includes connection health checks and more robust error handling.
    """

    def __init__(self):
        # Read environment variables for DB credentials
        self.db_host: str | None = os.getenv("host")
        port_env = os.getenv("port")
        self.db_port: int = int(port_env) if port_env else 5432
        self.db_name: str | None = os.getenv("database")
        self.db_user: str | None = os.getenv("db_user")
        self.db_password: str | None = os.getenv("password")
        self.sslmode: str | None = os.getenv("sslmode", "require")

        # Initialize the connection pool.
        # TCP keepalives let the server detect a dead client within ~90s
        # instead of waiting for OS defaults (~2h on macOS), so orphaned
        # connections from killed Python processes do not hold locks
        # on the server side for long.
        try:
            self.pool: ThreadedConnectionPool = ThreadedConnectionPool(
                minconn=1,
                maxconn=6,  # Adjust pool size as needed.
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                sslmode=self.sslmode,
                connect_timeout=15,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            logging.info("Database connection pool created successfully.")
        except Exception as e:
            logging.error("Error creating connection pool: %s", e)
            raise e

        self.timeout: int = 600  # Example timeout (10 minutes)
        self.last_connection_time: float = time.time()

    def get_connection(self):
        """Retrieve a connection from the pool and verify it is active.

        If the pooled connection is in an aborted-transaction state (e.g. a
        prior worker left it poisoned), we attempt a rollback first so the
        health-check SELECT 1 can succeed. If recovery fails we close the
        connection and tell the pool to drop it, then ask the pool for a
        fresh one. This prevents leaked connections that would otherwise
        exhaust the pool and deadlock all workers.
        """
        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed:
                logging.warning("Acquired a closed connection, requesting fresh one.")
                try:
                    self.pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self.pool.getconn()

            # Best-effort: clear any aborted-txn state inherited from prior worker.
            try:
                conn.rollback()
            except Exception:
                pass

            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            except Exception as hc_ex:
                logging.warning(
                    "Health-check failed (%s); discarding and retrying once.",
                    hc_ex,
                )
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    self.pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self.pool.getconn()
                try:
                    conn.rollback()
                except Exception:
                    pass
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return conn
        except Exception as e:
            logging.error("Error getting connection from pool: %s", e)
            # Avoid leaking a checked-out connection on the error path.
            if conn is not None:
                try:
                    self.pool.putconn(conn, close=True)
                except Exception:
                    pass
            return None

    def release_connection(self, conn: connection):
        """Return the connection to the pool safely.

        If the connection is closed or in a broken state, ask the pool to
        drop it instead of recycling so workers don't inherit poison.
        """
        if conn.closed:
            self.pool.putconn(conn, close=True)
            return
        try:
            # Belt-and-suspenders: clear any leftover txn state before recycling.
            try:
                conn.rollback()
            except Exception:
                # Connection is unusable; drop it.
                self.pool.putconn(conn, close=True)
                return
            self.pool.putconn(conn)
        except Exception as e:
            logging.error("Error releasing connection: %s", e)
            try:
                self.pool.putconn(conn, close=True)
            except Exception:
                pass

    def close_all_connections(self):
        """Closes all connections in the pool."""
        try:
            self.pool.closeall()
            logging.info("All pooled connections closed successfully.")
        except Exception as e:
            logging.error("Error closing connections: %s", e)

    def execute_query(self,
        sql: str | bytes,
        values: tuple[Any] | None = None,
        fetch: bool=False
    ) -> dict[str, str]:
        """
        Executes a query with optional parameters.
        If fetch=True, returns results.
        """
        conn = self.get_connection()
        if not conn:
            return {
                "status": "error",
                "message": "Could not obtain a database connection.",
            }

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql, values or ())
                if fetch:
                    result = cursor.fetchall()
                    conn.commit()
                    return {
                        "status": "success",
                        "message": "Query executed successfully.",
                        "data": str(result),
                    }
                conn.commit()
                return {"status": "success", "message": "Query executed successfully."}
        except Exception as e:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error("Rollback failed: %s", rollback_error)
            logging.error("Error executing query: %s", e)
            return {"status": "error", "message": str(e)}
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

    def bulk_inject_to_db(self,
        table,
        data: list[dict[Any, Any]],
        conflict_columns: list[str] = [""],
        schema: str = ""
    ) -> dict[str, str]:
        """
        Efficiently inserts multiple rows into a table using execute_values.
        Leverages 'ON CONFLICT DO NOTHING' if conflict_columns are provided.
        """
        if not data:
            return {"status": "success", "message": "No data to insert."}

        conn = self.get_connection()
        if not conn:
            return {
                "status": "error",
                "message": "Could not obtain a database connection.",
            }

        try:
            with conn.cursor() as cursor:
                columns = data[0].keys()
                schema_str = f"{schema}." if schema else ""

                sql = (
                    f"INSERT INTO {schema_str}{table} ({', '.join(columns)}) VALUES %s"
                )

                # Dynamically add the ON CONFLICT clause if conflict_columns are specified
                if conflict_columns:
                    sql += f" ON CONFLICT ({', '.join(conflict_columns)}) DO NOTHING"

                # Prepare data for execute_values
                values = [[row[col] for col in columns] for row in data]

                psycopg2.extras.execute_values(cursor, sql, values)
                inserted_count = cursor.rowcount
                conn.commit()

                return {
                    "status": "success",
                    "message": f"Successfully inserted or ignored {inserted_count} rows.",
                }

        except Exception as e:
            try:
                conn.rollback()
            except Exception as rollback_error:
                logging.error("Rollback failed: %s", rollback_error)
            logging.error("Error during bulk insert: %s", e)
            return {"status": "error", "message": str(e)}
        finally:
            self.release_connection(conn)

    def update_data(self,
        table,
        data,
        conditions: dict[Any, Any],
        schema: str = "public"
    ) -> dict[str, str]:
        """
        Updates records in a table based on provided conditions.
        """
        if not conditions:
            return {"status": "error", "message": "No conditions provided for update."}

        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"UPDATE {schema_str}{table} SET {set_clause} WHERE {where_clause}"
        values = list(data.values()) + list(conditions.values())
        return self.execute_query(sql, tuple(values))

    def delete_data(self,
        table,
        conditions: dict[Any, Any],
        schema: str = "public"
    ) -> dict[str, str]:
        """
        Deletes records matching the provided conditions.
        """
        if not conditions:
            return {
                "status": "error",
                "message": "No conditions provided for deletion.",
            }

        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        schema_str = f"{schema}." if schema else ""
        sql = f"DELETE FROM {schema_str}{table} WHERE {where_clause}"
        return self.execute_query(sql, tuple(conditions.values()))

    def read_db(self,
        table=None,
        columns="*",
        conditions=None,
        schema=None,
        sql=None
    ) -> dict[str, str]:
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
            where_clause: str = " AND ".join([f"{col} = %s" for col in conditions.keys()])
            query += f" WHERE {where_clause}"
            values = tuple(conditions.values())
        return self.execute_query(query, values, fetch=True)
