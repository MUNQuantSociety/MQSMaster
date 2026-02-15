import logging

try:
    from common.database.MQSDBConnector import MQSDBConnector
except ImportError:
    logging.warning("MQSDBConnector relative import failed; using absolute import.")
    from src.common.database.MQSDBConnector import MQSDBConnector


class SchemaDefinitions:
    def __init__(self):
        self.db = MQSDBConnector()

    def create_all_tables(self):
        res = self.db.execute_query("SELECT 1", fetch=True)
        if res["status"] == "error":
            print("Error connecting to DB:", res["message"])
            return

        create_user_creds_table = """
        CREATE TABLE IF NOT EXISTS user_creds (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        );
        """

        # MATCHES SERVER: column name is `timestamp`
        create_market_data_table = """
        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            date DATE NOT NULL,
            exchange VARCHAR(50) NOT NULL,
            open_price NUMERIC,
            high_price NUMERIC,
            low_price NUMERIC,
            close_price NUMERIC,
            volume BIGINT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        # MATCHES SERVER: column name is `exec_timestamp`
        create_trade_logs_table = """
        CREATE TABLE IF NOT EXISTS trade_execution_logs (
            trade_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            ticker VARCHAR(10),
            exec_timestamp TIMESTAMPTZ NOT NULL,
            side VARCHAR(4) NOT NULL,
            quantity NUMERIC NOT NULL,
            arrival_price NUMERIC NOT NULL,
            exec_price NUMERIC NOT NULL,
            slippage_bps NUMERIC,
            notional NUMERIC,
            notional_local NUMERIC,
            currency VARCHAR(10),
            fx_rate NUMERIC,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        # MATCHES SERVER: column name is `timestamp`
        create_pnl_book_table = """
        CREATE TABLE IF NOT EXISTS pnl_book (
            pnl_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            timestamp TIMESTAMPTZ NOT NULL,
            date DATE NOT NULL,
            realized_pnl NUMERIC,
            unrealized_pnl NUMERIC,
            fx_rate NUMERIC,
            currency VARCHAR(10),
            notional NUMERIC,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        # MATCHES SERVER: column name is `timestamp`
        create_risk_book_table = """
        CREATE TABLE IF NOT EXISTS risk_book (
            risk_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50),
            date DATE NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            risk_metric VARCHAR(100),
            value NUMERIC,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        # MATCHES SERVER: column name is `timestamp`
        create_cash_equity_book_table = """
        CREATE TABLE IF NOT EXISTS cash_equity_book (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            date DATE NOT NULL,
            portfolio_id VARCHAR(50) NOT NULL,
            currency VARCHAR(10) NOT NULL,
            notional NUMERIC NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        create_positions_table = """
        CREATE TABLE IF NOT EXISTS positions_book (
            position_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50) NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            quantity NUMERIC NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (portfolio_id, ticker)
        );
        """

        create_port_weights_table = """
        CREATE TABLE IF NOT EXISTS portfolio_weights (
            weights_id SERIAL PRIMARY KEY,
            portfolio_id VARCHAR(50) NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            weight NUMERIC NOT NULL,
            model VARCHAR(50),
            date DATE NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (portfolio_id, ticker, date, model)
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
            create_port_weights_table,
        ]

        for stmt in statements:
            result = self.db.execute_query(stmt)
            if result["status"] == "error":
                print("Error creating table:", result["message"])
                return

        print("All tables created or confirmed to exist.")
