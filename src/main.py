# main.py
# Entry point for the MQS Trading System.
# main.py is reserved for live trading purposes.


import logging

try:
    from common.database.MQSDBConnector import MQSDBConnector
    from live_trading.engine import RunEngine
    from live_trading.executor import tradeExecutor
    from portfolios.portfolio_1.strategy import VolMomentum
    from portfolios.portfolio_2.strategy import MomentumStrategy
    from portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
    from portfolios.portfolio_dummy.strategy import CrossoverRmiStrategy
except ImportError:
    logging.debug("Necessary modules relative imports failed; using absolute import.")
    from src.common.database.MQSDBConnector import MQSDBConnector
    from src.live_trading.engine import RunEngine
    from src.live_trading.executor import tradeExecutor
    from src.portfolios.portfolio_1.strategy import VolMomentum
    from src.portfolios.portfolio_2.strategy import MomentumStrategy
    from src.portfolios.portfolio_3.strategy import RegimeAdaptiveStrategy
    from src.portfolios.portfolio_dummy.strategy import CrossoverRmiStrategy

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main entry point for the MQS Trading System.
    comment/uncomment portfolio classes in the portfolio_classes list to run different strategies.
    """
    portfolio_classes = [
        VolMomentum,
        MomentumStrategy
    ]

    # DO NOT CHANGE BELOW THIS LINE
    # ======================================================
    db_conn = None
    try:
        db_conn = MQSDBConnector()
        logging.info("Database connector initialized.")

        # Ensure all tables exist. Idempotent -- uses CREATE TABLE IF NOT EXISTS.
        # Defends against running this entrypoint without start.sh having run schema
        # bootstrap via rbp_runner.
        try:
            try:
                from common.database.schemaDefinitions import SchemaDefinitions
            except ImportError:
                from src.common.database.schemaDefinitions import SchemaDefinitions
            SchemaDefinitions().create_all_tables()
            logging.info("Schema bootstrap complete.")
        except Exception as exc:
            logging.exception("Schema bootstrap failed; continuing anyway: %s", exc)

        # Load rbp_overlay config from master config.
        import json
        from pathlib import Path
        master_cfg_path = Path(__file__).resolve().parent / "portfolios" / "portfolio_manager_config.json"
        try:
            with open(master_cfg_path) as f:
                master_cfg = json.load(f)
            rbp_overlay_cfg = master_cfg.get("rbp_overlay", {"enabled": False})
        except Exception as exc:
            logging.warning("Failed to load rbp_overlay config (%s); overlay disabled.", exc)
            rbp_overlay_cfg = {"enabled": False}

        rbp_overlay = None
        if rbp_overlay_cfg.get("enabled", False):
            try:
                from risk_manager.rbp_overlay import RBPOverlay
            except ImportError:
                from src.risk_manager.rbp_overlay import RBPOverlay
            rbp_overlay = RBPOverlay(db_conn, rbp_overlay_cfg)
            logging.info("RBPOverlay constructed.")

        live_executor = tradeExecutor(db_conn, rbp_overlay=rbp_overlay)
        logging.info("Live executor initialized.")

        run_engine = RunEngine(db_connector=db_conn, executor=live_executor)
        logging.info("Run engine initialized.")

        run_engine.load_portfolios(portfolio_classes)
        logging.info("Run engine setup complete.")

        run_engine.run()

    except Exception as e:
        logging.critical(
            f"A critical error occurred in the main application loop: {e}",
            exc_info=True,
        )
    finally:
        logging.info("MQS Trading System is shutting down.")


if __name__ == "__main__":
    main()
