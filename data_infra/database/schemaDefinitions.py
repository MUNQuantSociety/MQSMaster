"""
schemaDefinitions.py
--------------------
Contains SQL DDL or other schema setup for the 'Master market data warehouse'
and related tables (Trade Execution Logs, PnL Book, Risk Book, etc.).
You can call these definitions from your main setup script or orchestrator
to create required tables in PostgreSQL.
"""

from data_infra.database.MQSDBConnector import MQSDBConnector

class SchemaDefinitions:
    """
    Encapsulates methods to create or drop tables in your PostgreSQL database.
    Adjust the CREATE TABLE statements to match your real schema needs.
    """

    def __init__(self):
        self.db = MQSDBConnector()

    def create_all_tables(self):
        """
        Create all necessary tables (if they do not already exist).
        """
        res = self.db.connect()
        if res['status'] == 'error':
            print("Error connecting to DB:", res['message'])
            return
        
        create_user_creds_table = """
        CREATE TABLE IF NOT EXISTS user_creds (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        );
        """
        
        # Example DDL statements:
        create_market_data_table = """
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            trade_time TIMESTAMP NOT NULL,
            open_price NUMERIC,
            high_price NUMERIC,
            low_price NUMERIC,
            close_price NUMERIC,
            volume BIGINT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_trade_logs_table = """
        CREATE TABLE IF NOT EXISTS trade_execution_logs (
            trade_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            ticker VARCHAR(10),
            execution_time TIMESTAMP NOT NULL,
            side VARCHAR(4) NOT NULL,  -- e.g. 'BUY' or 'SELL'
            quantity NUMERIC NOT NULL,
            price NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_pnl_book_table = """
        CREATE TABLE IF NOT EXISTS pnl_book (
            pnl_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            date DATE NOT NULL,
            realized_pnl NUMERIC,
            unrealized_pnl NUMERIC,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_risk_book_table = """
        CREATE TABLE IF NOT EXISTS risk_book (
            risk_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            date DATE NOT NULL,
            risk_metric VARCHAR(100),
            value NUMERIC,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_live_positions_table = """
        CREATE TABLE IF NOT EXISTS live_positions_book (
            position_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            ticker VARCHAR(10),
            quantity NUMERIC NOT NULL,
            avg_cost NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        statements = [
            create_user_creds_table,
            create_market_data_table,
            create_trade_logs_table,
            create_pnl_book_table,
            create_risk_book_table,
            create_live_positions_table
        ]

        for stmt in statements:
            try:
                self.db.cursor.execute(stmt)
            except Exception as e:
                print("Error creating table:", e)

        self.db.connection.commit()
        print("All tables created or confirmed to exist.")

    def drop_all_tables(self):
        """
        If needed, drop all tables (Dangerous in production).
        """
        res = self.db.check_connection()
        if res['status'] == 'error':
            print("Error connecting to DB:", res['message'])
            return

        drop_statements = [
            "DROP TABLE IF EXISTS market_data;",
            "DROP TABLE IF EXISTS trade_execution_logs;",
            "DROP TABLE IF EXISTS pnl_book;",
            "DROP TABLE IF EXISTS risk_book;",
            "DROP TABLE IF EXISTS live_positions_book;"
        ]

        for stmt in drop_statements:
            try:
                self.db.cursor.execute(stmt)
            except Exception as e:
                print("Error dropping table:", e)

        self.db.connection.commit()
        print("All tables dropped.")
