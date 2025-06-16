# new.py
import os
import json
from typing import Dict, Any
from data_infra.database.MQSDBConnector import MQSDBConnector

BASE_PORTFOLIO_PATH = os.path.join(os.path.dirname(__file__), "portfolios")


def _config_path(portfolio_id: str) -> str:
    """Return the path to this portfolio's config.json."""
    dirname = f"portfolio_{portfolio_id}"
    return os.path.join(BASE_PORTFOLIO_PATH, dirname, "config.json")


def fetch_portfolio_details(portfolio_id: str) -> Dict[str, Any]:
    """
    Returns a dict with:
      - portfolio_id (str)
      - tickers (List[str])
      - weights (Dict[str, float])
      - portfolio_notional (float)
    """
    # 1) load static config
    cfg_path = _config_path(portfolio_id)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json for portfolio {portfolio_id}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 2) fetch latest notional from DB
    db = MQSDBConnector()
    sql = """
      SELECT notional
      FROM pnl_book
      WHERE portfolio_id = %s
      ORDER BY timestamp DESC
      LIMIT 1
    """
    res = db.execute_query(sql, values=(portfolio_id,), fetch=True)
    if res["status"] != "success" or not res["data"]:
        raise RuntimeError("Failed to fetch portfolio notional")
    notional = res["data"][0]["notional"]

    return {
        "portfolio_id": cfg["PORTFOLIO_ID"],
        "tickers": cfg["TICKERS"],
        "weights": cfg.get("WEIGHTS", {}),
        "portfolio_notional": notional,
    }


def update_ticker_weight(portfolio_id: str, ticker: str, new_weight: float) -> Dict[str, float]:
    """
    Sets `ticker` to `new_weight` (0–1) and scales all other tickers so sum(weights)=1.
    Writes the updated WEIGHTS back to config.json and returns the new weights dict.
    """
    details = fetch_portfolio_details(portfolio_id)
    weights = details["weights"].copy()
    if ticker not in weights:
        raise KeyError(f"{ticker} is not in portfolio {portfolio_id}")

    # assign new weight
    weights[ticker] = new_weight

    # re-normalize the others to fill the remainder
    others = [t for t in weights if t != ticker]
    total_others = sum(weights[t] for t in others)
    remainder = 1.0 - new_weight
    if total_others <= 0:
        # if it was the only ticker or others sum to zero, split remainder equally
        for t in others:
            weights[t] = remainder / len(others)
    else:
        scale = remainder / total_others
        for t in others:
            weights[t] *= scale

    # persist back to config.json
    cfg_path = _config_path(portfolio_id)
    with open(cfg_path, "r+") as f:
        cfg = json.load(f)
        cfg["WEIGHTS"] = weights
        f.seek(0)
        json.dump(cfg, f, indent=2)
        f.truncate()

    return weights


def allocate_notional(portfolio_id: str) -> Dict[str, float]:
    """
    Fetches available cash for `portfolio_id` and returns
    a mapping { ticker: cash_amount = total_cash * weight }.
    """
    # 1) fetch cash equity
    db = MQSDBConnector()
    sql = """
      SELECT notional
      FROM cash_equity_book
      WHERE portfolio_id = %s
      ORDER BY timestamp DESC
      LIMIT 1
    """
    res = db.execute_query(sql, values=(portfolio_id,), fetch=True)
    if res["status"] != "success" or not res["data"]:
        raise RuntimeError("Failed to fetch cash equity")
    total_cash = res["data"][0]["notional"]

    # 2) get up-to-date weights
    details = fetch_portfolio_details(portfolio_id)
    weights = details["weights"]

    # 3) allocate
    allocations = {t: total_cash * w for t, w in weights.items()}
    return allocations

def main():
    details = fetch_portfolio_details("2")
    print(json.dumps(details, indent=2))

if __name__ == "__main__":
    main()
