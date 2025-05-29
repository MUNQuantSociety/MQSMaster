from data_infra.database.MQSDBConnector import MQSDBConnector

class SchemaDefinitions:
    """
    DO NOT CALL drop_all_tables() PLEASE!
    Encapsulates methods to create or drop tables in the MQS PostgreSQL database.
    Adjust the CREATE TABLE statements to match your real schema needs.
    """

    def __init__(self):
        self.db = MQSDBConnector()

    def create_all_tables(self):
        """
        Create all necessary tables (if they do not already exist).
        """
        # Test connection by executing a simple query
        res = self.db.execute_query("SELECT 1", fetch=True)
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
        
        create_market_data_table = """
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            date DATE NOT NULL,
            exchange VARCHAR(50) NOT NULL,
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
            exec_timestamp TIMESTAMP NOT NULL,
            side VARCHAR(4) NOT NULL,  -- e.g. 'BUY' or 'SELL'
            quantity NUMERIC NOT NULL,
            price_last NUMERIC NOT NULL,
            notional NUMERIC,
            notional_local NUMERIC,
            currency VARCHAR(10),
            fx_rate NUMERIC,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        create_pnl_book_table = """
        CREATE TABLE IF NOT EXISTS pnl_book (
            pnl_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            timestamp TIMESTAMP NOT NULL,
            date DATE NOT NULL,
            realized_pnl NUMERIC,
            unrealized_pnl NUMERIC,
            fx_rate NUMERIC,
            currency VARCHAR(10),
            notional NUMERIC,
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

        create_cash_equity_book_table = """
        CREATE TABLE IF NOT EXISTS cash_equity_book (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            date DATE NOT NULL,
            portfolio_id VARCHAR(50) NOT NULL,
            currency VARCHAR(10) NOT NULL,
            notional NUMERIC NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        create_positions_table = """
        CREATE TABLE IF NOT EXISTS positions (
            position_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50) NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            quantity NUMERIC NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (portfolio_id, ticker)
        );
        """
        create_port_weights_table = """
        CREATE TABLE IF NOT EXISTS portfolio_weights (
        weights_id SERIAL PRIMARY KEY,
        portfolio_id VARCHAR(50) NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        weight NUMERIC NOT NULL, 
        model VARCHAR(50), -- version or name of the model used to calculate the weight
        date DATE NOT NULL,
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (portfolio_id, ticker, date, model) -- Ensures one weight per asset, per portfolio, per day, per model
        );
        """

        statements = [
            create_user_creds_table,
            create_market_data_table,
            create_trade_logs_table,
            create_pnl_book_table,
            create_risk_book_table,
            create_cash_equity_book_table,
            create_positions_table,
            create_port_weights_table
        ]

        for stmt in statements:
            result = self.db.execute_query(stmt)
            if result['status'] == 'error':
                print("Error creating table:", result['message'])
        print("All tables created or confirmed to exist.")

    def drop_all_tables(self):
        """
        If needed, drop all tables (Dangerous in production).
        """
        # Test connection similarly
        res = self.db.execute_query("SELECT 1", fetch=True)
        if res['status'] == 'error':
            print("Error connecting to DB:", res['message'])
            return

        drop_statements = [
            "DROP TABLE IF EXISTS market_data;",
            "DROP TABLE IF EXISTS trade_execution_logs;",
            "DROP TABLE IF EXISTS pnl_book;",
            "DROP TABLE IF EXISTS risk_book;",
            "DROP TABLE IF EXISTS cash_equity_book;"
        ]

        for stmt in drop_statements:
            result = self.db.execute_query(stmt)
            if result['status'] == 'error':
                print("Error dropping table:", result['message'])
        print("All tables dropped.")