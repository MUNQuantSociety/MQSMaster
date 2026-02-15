import os
import time
import logging
from typing import Dict, List, Optional, Sequence, Any

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv


# Configure logging for better debugging and tracing.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Load environment variables from .env (if present)
load_dotenv()


class MQSDBConnector:
    """
    Thread-safe PostgreSQL connector class using connection pooling.
    Includes connection health checks and robust error handling.
    """

    def __init__(self):
        # Prefer standard DB_* env vars, but support legacy keys used in this repo.
        self.db_host = os.getenv("DB_HOST") or os.getenv("host")
        self.db_port = int(os.getenv("DB_PORT") or os.getenv("port") or 5432)
        self.db_name = os.getenv("DB_NAME") or os.getenv("database")
        self.db_user = os.getenv("DB_USER") or os.getenv("db_user")
        self.db_password = os.getenv("DB_PASSWORD") or os.getenv("password")
        self.sslmode = os.getenv("DB_SSLMODE") or os.getenv("sslmode") or "prefer"

        # Basic validation so you fail fast with a useful error.
        missing = []
        if not self.db_host:
            missing.append("DB_HOST/host")
        if not self.db_name:
            missing.append("DB_NAME/database")
        if not self.db_user:
            missing.append("DB_USER/db_user")
        if not self.db_password:
            missing.append("DB_PASSWORD/password")

        if missing:
            raise ValueError(f"Missing required DB environment variables: {', '.join(missing)}")

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
                sslmode=self.sslmode,
            )
            logging.info("Database connection pool created successfully.")
        except Exception as e:
            logging.error("Error creating connection pool: %s", e)
            raise

        self.timeout = 600  # Example timeout (10 minutes)
        self.last_connection_time = time.time()

    # ---------- internal helpers ----------

    @staticmethod
    def _quote_ident(name: str) -> str:
        """Safely quote a SQL identifier (table/column)."""
        # Double-quote and escape embedded quotes.
        return '"' + name.replace('"', '""') + '"'

    @classmethod
    def _quote_table(cls, table: str, schema: Optional[str] = None) -> str:
        """Quote schema/table properly."""
        if schema:
            return f"{cls._quote_ident(schema)}.{cls._quote_ident(table)}"
        return cls._quote_ident(table)

    # ---------- pool management ----------

    def get_connection(self):
        """Retrieve a connection from the pool and verify it is active."""
        try:
            conn = self.pool.getconn()

            # If connection is closed, return it to pool (closing it), then get a new one.
            if conn is None or conn.closed:
                logging.warning("Acquired a closed connection, replacing it...")
                if conn is not None:
                    try:
                        self.pool.putconn(conn, close=True)
                    except Exception:
                        pass
                conn = self.pool.getconn()

            # Health check
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

    # ---------- query execution ----------

    def execute_query(self, sql: str, values: Optional[Sequence[Any]] = None, fetch: bool = False):
        """
        Executes a query with optional parameters.
        If fetch=True, returns results.
        """
        conn = self.get_connection()
        if not conn:
            return {"status": "error", "message": "Could not obtain a database connection."}

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql, values or ())
                if fetch:
                    result = cursor.fetchall()
                    conn.commit()
                    return {"status": "success", "message": "Query executed successfully.", "data": result}
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

    # ---------- inserts ----------

    def inject_to_db(self, table: str, data: Dict, schema: Optional[str] = None):
        """
        Inserts a single row into the specified table.
        Quotes identifiers to avoid issues with reserved words (e.g., timestamp).
        """
        if not data:
            return {"status": "error", "message": "No data provided for insert."}

        columns = list(data.keys())
        col_sql = ", ".join([self._quote_ident(c) for c in columns])
        placeholders = ", ".join(["%s"] * len(columns))
        table_sql = self._quote_table(table, schema=schema)

        sql = f"INSERT INTO {table_sql} ({col_sql}) VALUES ({placeholders})"
        return self.execute_query(sql, tuple(data[c] for c in columns))

    def bulk_inject_to_db(
        self,
        table: str,
        data: List[Dict],
        conflict_columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
    ):
        """
        Efficiently inserts multiple rows into a table using execute_values.
        Leverages 'ON CONFLICT DO NOTHING' if conflict_columns are provided.
        Quotes identifiers and validates consistent keys across rows.
        """
        if not data:
            return {"status": "success", "message": "No data to insert."}

        # Lock column order from first row
        columns = list(data[0].keys())
        if not columns:
            return {"status": "error", "message": "First row has no columns."}

        # Validate all rows have same keys (at least the required set)
        for i, row in enumerate(data):
            missing = set(columns) - set(row.keys())
            if missing:
                return {"status": "error", "message": f"Row {i} missing keys: {sorted(missing)}"}

        conn = self.get_connection()
        if not conn:
            return {"status": "error", "message": "Could not obtain a database connection."}

        try:
            with conn.cursor() as cursor:
                table_sql = self._quote_table(table, schema=schema)
                col_sql = ", ".join([self._quote_ident(c) for c in columns])
                sql = f"INSERT INTO {table_sql} ({col_sql}) VALUES %s"

                if conflict_columns:
                    conflict_sql = ", ".join([self._quote_ident(c) for c in conflict_columns])
                    sql += f" ON CONFLICT ({conflict_sql}) DO NOTHING"

                values = [[row[c] for c in columns] for row in data]

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

    # ---------- updates / deletes / reads ----------

    def update_data(self, table: str, data: Dict, conditions: Optional[Dict] = None, schema: Optional[str] = None):
        """
        Updates records in a table based on provided conditions.
        Quotes identifiers.
        """
        if not conditions:
            return {"status": "error", "message": "No conditions provided for update."}
        if not data:
            return {"status": "error", "message": "No data provided for update."}

        set_clause = ", ".join([f"{self._quote_ident(col)} = %s" for col in data.keys()])
        where_clause = " AND ".join([f"{self._quote_ident(col)} = %s" for col in conditions.keys()])
        table_sql = self._quote_table(table, schema=schema)

        sql = f"UPDATE {table_sql} SET {set_clause} WHERE {where_clause}"
        values = list(data.values()) + list(conditions.values())
        return self.execute_query(sql, tuple(values))

    def delete_data(self, table: str, conditions: Optional[Dict] = None, schema: Optional[str] = None):
        """
        Deletes records matching the provided conditions.
        Quotes identifiers.
        """
        if not conditions:
            return {"status": "error", "message": "No conditions provided for deletion."}

        where_clause = " AND ".join([f"{self._quote_ident(col)} = %s" for col in conditions.keys()])
        table_sql = self._quote_table(table, schema=schema)

        sql = f"DELETE FROM {table_sql} WHERE {where_clause}"
        return self.execute_query(sql, tuple(conditions.values()))

    def read_db(
        self,
        table: Optional[str] = None,
        columns: str = "*",
        conditions: Optional[Dict] = None,
        schema: Optional[str] = None,
        sql: Optional[str] = None,
    ):
        """
        Retrieves data from the database.
        If a custom SQL is provided, it will be executed directly.
        If table/conditions are used, identifiers are quoted.
        """
        if sql:
            return self.execute_query(sql, fetch=True)

        if not table:
            return {"status": "error", "message": "Table name must be provided if no custom SQL is used."}

        table_sql = self._quote_table(table, schema=schema)
        query = f"SELECT {columns} FROM {table_sql}"
        values = None

        if conditions:
            where_clause = " AND ".join([f"{self._quote_ident(col)} = %s" for col in conditions.keys()])
            query += f" WHERE {where_clause}"
            values = tuple(conditions.values())

        return self.execute_query(query, values, fetch=True)
