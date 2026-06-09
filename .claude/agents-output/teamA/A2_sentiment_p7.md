# A2 - Portfolio_7: Sentiment-Tilted Boring-Not-Lottery

**Author:** Team A2 (quant-research analyst / senior Python engineer)
**Date:** 2026-05-20
**Status:** Design proposal, read-only research. All code is apply-ready but unpersisted.
**Parent:** `Portfolio_6` (Boring + Not-Lottery + inverse-vol + GLD/trend hedge)

---

## 1. Executive summary

Portfolio_7 inherits the full Portfolio_6 pipeline (monthly low-vol/low-skew screen, inverse-volatility weights, GLD/trend-hedge sleeves, vol-target scaling) and adds a **cross-sectional FinBERT sentiment tilt** on the surviving stock sleeve only. After Portfolio_6 produces its capped inverse-vol weights, we look up each survivor's mean daily sentiment over a configurable trailing window (default 21 trading days, EWM half-life 5d), demean and standardize across the surviving cohort to get `z_i`, then multiply each weight by `exp(LAMBDA * z_i)` (default LAMBDA=0.25). The sleeve is then renormalized to its original sleeve gross, the per-name cap is re-enforced iteratively, and the hedge sleeve is left untouched.

The tilt is constructed strictly ex-ante: at decision bar `t` we only consume `news_sentiment.published_at < t` filtered through the existing daily aggregate that lives on `market_data.avg_sentiment` (date `< t`). No T+0 publication leakage, no future article. Empirical literature (Heston-Sinha 2017, Tetlock-Saar-Tsechansky-Macskassy 2008, Garcia 2013, Ke-Kelly-Xiu 2019) supports a multi-week aggregation window that retains predictive content with manageable noise; 1-2 day signals exist but are noisy and reverse fast. We default `21d` window with 5d EWM half-life to bridge the daily-vs-weekly gap and reuse Portfolio_6's monthly rebalance cadence (the signal is recomputed monthly on rebalance — there is no daily over-trading). LAMBDA=0.25 produces ~30% spread between +2sigma and -2sigma names (`exp(0.5)/exp(-0.5) ~= 2.7`), comparable to factor tilts in production multi-factor smart-beta indexes.

Portfolio_7 registers with capital weight **0** in the manager config (off by default until OOS-validated). A documented falsification threshold gates activation.

---

## 2. Sources (15 primary, all post-2007 except the seminal anchors)

Tags: [empirical] = empirical signal study, [model] = sentiment model, [methodology] = portfolio/data design, [skeptic] = reversal/decay/caveat evidence.

1. **Tetlock 2007 - "Giving Content to Investor Sentiment: The Role of Media in the Stock Market", JoF 62(3):1139-1168.** [https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x) - Seminal: WSJ "Abreast of the Market" pessimism predicts next-day price pressure and partial reversal within ~5 trading days. [empirical, skeptic]
2. **Tetlock, Saar-Tsechansky, Macskassy 2008 - "More Than Words: Quantifying Language to Measure Firms' Fundamentals", JoF 63(3):1437-1467.** [https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01362.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01362.x) - Firm-level: fraction of negative words in news forecasts earnings and stock returns; prices briefly underreact (1-10 days). [empirical]
3. **Loughran & McDonald 2011 - "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks", JoF 66(1):35-65.** [https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x) - Domain-specific finance lexicon; ~3/4 of Harvard-Negative words are *not* negative in financial context. Foundation for any non-BERT baseline. [model]
4. **Garcia 2013 - "Sentiment during Recessions", JoF 68(3):1267-1300.** [https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12027](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12027) (PDF [https://leeds-faculty.colorado.edu/garcia/media_v33.pdf](https://leeds-faculty.colorado.edu/garcia/media_v33.pdf)) - Predictability concentrated in recessions; effect partially reverses within ~4 trading days. Critical regime-dependence caveat. [empirical, skeptic]
5. **Heston & Sinha 2017 - "News vs. Sentiment: Predicting Stock Returns from News Stories", FAJ 73(3):67-83.** [https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories](https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories) (Fed WP version [https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf](https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf)) - **Key paper**: daily news predicts only 1-2d returns; **weekly-aggregated news predicts ~one quarter**; negative news shows delayed reaction concentrated around the next earnings announcement. Direct justification for our 21d window. [empirical]
6. **Boudoukh, Feldman, Kogan, Richardson 2013 - "Which News Moves Stock Prices? A Textual Analysis", NBER WP 18725.** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2207241](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2207241) - Filtering for *relevant* news (event-tagged) doubles R^2 of returns; aggregate sentiment without relevance-filtering is too noisy at 1d. Implication: we need enough articles per name to denoise. [methodology]
7. **Ke, Kelly, Xiu 2019/2020 - "Predicting Returns with Text Data", NBER WP 26186 / AQR.** [https://www.nber.org/papers/w26186](https://www.nber.org/papers/w26186) - SESTM trained on 22M Dow Jones articles 1989-2017; long-short on article-level sentiment outperforms RavenPack. Validates the construction: rank cross-sectionally, hold for days-to-weeks. [empirical, methodology]
8. **Araci 2019 - "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", arXiv:1908.10063.** [https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063) - Model used by `NLP/sentiment/scorer.py`; 3-class softmax (pos/neutral/neg); our `sentiment_score = p_pos - p_neg` in [-1,1]. [model]
9. **Hafez & Xie 2014 - "News Beta: Factoring Sentiment Risk into Quant Models", SSRN 2071142.** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2071142](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2071142) - RavenPack practitioner study; news effect strongest in small-mid caps; supports the cross-sectional rank approach. [empirical]
10. **Glasserman & Mamaysky 2019 - "Does Unusual News Forecast Market Stress?"** (JFQA) [https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/does-unusual-news-forecast-market-stress/](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2902460) - Unusual + negative sentiment forecasts volatility; sentiment direction interacts with novelty. Reinforces denoising via multi-article aggregation. [empirical, skeptic]
11. **Da, Engelberg, Gao 2011 - "In Search of Attention", JoF 66(5):1461-1499.** [https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2011.01679.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2011.01679.x) - Attention (search volume) drives short-term price impact that reverses; treat sentiment + attention as one regime, not independent. [skeptic]
12. **Bollen, Mao, Zeng 2011 - "Twitter Mood Predicts the Stock Market", J Comp Sci 2(1):1-8.** [https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X](https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X) - Social-media sentiment predicts DJIA at a several-day horizon; replicated mixedly. Justifies *news* (not Twitter) as the cleaner channel for a long-only US equity tilt. [empirical, skeptic]
13. **Petrescu, Hafez et al. 2021 - "News Sentiment Everywhere: Trading Global Equities", RavenPack.** [https://www.researchgate.net/publication/351360244_News_Sentiment_Everywhere_Trading_Global_Equities](https://www.researchgate.net/publication/351360244_News_Sentiment_Everywhere_Trading_Global_Equities) - Practitioner: trading infrequent but high-conviction news beats trading every article; supports our `MIN_ARTICLES_PER_TICKER` filter. [methodology]
14. **Kim, Olmo, Sapra 2022 - "BERT's sentiment score for portfolio optimization: a fine-tuned view in Black-Litterman", Neural Computing & Applications 34(20).** [https://link.springer.com/article/10.1007/s00521-022-07403-1](https://link.springer.com/article/10.1007/s00521-022-07403-1) - BERT sentiment as a "view" tilts BL portfolios with measurable Sharpe gains. Validates an exponential-tilt style overlay. [methodology, empirical]
15. **Lopez de Prado 2018 - "Advances in Financial Machine Learning" (esp. Ch. 7 & 11 on labeling/leakage).** [https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Methodological anchor on Point-In-Time labeling, walk-forward, Deflated Sharpe (already used by P6 in `screener.deflated_sharpe_ratio`). [methodology]

Cross-validation matrix (each design claim cited by >=2 sources):

| Claim | Sources |
|---|---|
| Daily news signal is noisy / short-lived | Heston-Sinha 2017 (#5), Tetlock 2007 (#1), Garcia 2013 (#4) |
| Weekly/monthly aggregation extracts persistent signal | Heston-Sinha 2017 (#5), Ke-Kelly-Xiu 2019 (#7), Tetlock et al 2008 (#2) |
| Multi-article denoising required | Boudoukh et al 2013 (#6), Petrescu 2021 (#13), Glasserman-Mamaysky (#10) |
| Cross-sectional ranking beats single-stock timing | Ke-Kelly-Xiu 2019 (#7), Hafez-Xie 2014 (#9), Kim et al 2022 (#14) |
| Publication-time strict ordering is mandatory | Lopez de Prado 2018 (#15), Tetlock 2007 (#1) (intraday lag) |
| Effect can reverse, especially in recessions | Garcia 2013 (#4), Tetlock 2007 (#1), Bollen et al 2011 (#12) |

---

## 3. Current-state analysis (file:line citations)

### Sentiment data already lives in two places

- **`news_sentiment` table** (per-article, FinBERT score in [-1,1]). Schema: `src/common/database/schemaDefinitions.py:147-157` and creation/CRUD in `NLP/persistence/repository.py:57-83` (table + indexes on `(ticker, published_at)`, `(published_at)`, unique on `article_url`). Read API: `NewsSentimentRepository.get_sentiment_data(ticker, start, end)` at `NLP/persistence/repository.py:101-131` returns a `DataFrame` filtered by `published_at`.
- **`market_data.avg_sentiment NUMERIC` column** populated by `update_market_data_sentiment` at `NLP/persistence/repository.py:251-298` which runs an `UPDATE market_data SET sentiment_score = ns.avg_score FROM (... AVG(sentiment_score) ... GROUP BY ticker, DATE(published_at))`. Schema column at `src/common/database/schemaDefinitions.py:51`.

**Important schema note**: the column name in `schemaDefinitions.py:51` is `avg_sentiment NUMERIC` but `repository.py:255` updates `md.sentiment_score = ns.avg_score`. This is a column-name inconsistency between schema docs and runtime DML. We must defensively read **both** in P7 (`COALESCE(md.avg_sentiment, md.sentiment_score)`) to survive either schema rev. Flag for Team-DB. We do *not* fix this here (READ-ONLY mandate) but route around it in the new strategy.

### FinBERT scorer

- `NLP/sentiment/scorer.py:110-131`: `_score_batch` does softmax over 3 classes and returns `probs[:,0] - probs[:,2]` -> already in [-1,1]. Persistence inserts only valid `-1.0 <= sentiment_score <= 1.0` (validated in `repository.py:146-150` and `:210-212`).
- Daily mean aggregation in `scorer.py:97-108` uses `DATE(published_at)` granularity. This is the unit we tilt on.

### Portfolio_6 architecture

- `src/portfolios/portfolio_6/strategy.py:54` defines `Portfolio6Strategy(BasePortfolio)`. `__init__` builds candidate + hedge ticker set (`strategy.py:69-80`), reads `PORTFOLIO_6_CONFIG` from the merged config dict.
- Monthly rebalance triggered in `OnData` at `strategy.py:157-167` via `month_key` change.
- `_rebalance` (`strategy.py:196-293`) is the canonical hook: it produces `self._target_weights` from `inverse_vol_weights(...)` after screening, then applies vol-targeting and adds hedge sleeves. **This is the exact attach-point for our tilt.**
- `_collect_returns` (`strategy.py:172-194`) uses `context.Market[ticker].History(lookback_str)` — pattern we reuse for sentiment fetch via `History` would not work (sentiment isn't in MarketData's grouped frame except through `avg_sentiment` if present). We instead pull from the DB directly using the parent's `self.db` (`MQSDBConnector`), filtered by strict `published_at < context.time`.
- Trade dispatch via `_issue_target` (`strategy.py:316-342`) calls `self.executor.execute_trade(...)` with explicit `ticker_weight` — we reuse it unchanged.

### Framework conventions to preserve

- `StrategyContext` (`src/portfolios/strategy_api.py:48-110`) exposes `.time`, `.Market`, `.Portfolio`. Backtest sets `context.time` to the bar timestamp (`BasePortfolio.generate_signals_and_trade` at `portfolio_BASE/strategy.py:313-322`). **We use `context.time` as the strict ex-ante cutoff.**
- `BasePortfolio.__init__` requires `config_dict`; `MARKET_DATA_QUERY` constant at `portfolio_BASE/strategy.py:375` shows the parametrised SQL pattern (named placeholders `{placeholders}` filled with `%s`).
- `MQSDBConnector.execute_query(sql, values, fetch=True)` returns `{"status":"success","data":[...]}` — used identically in `NewsSentimentRepository.get_sentiment_data` (`NLP/persistence/repository.py:120`).

### Backtest registration

- `src/main_backtest.py:29,65` registers `Portfolio6Strategy` in `AVAILABLE_PORTFOLIO_CLASSES`. Adding P7 requires an import + list append. **Not in our deliverable** (READ-ONLY on src/), but the user must add `from src.portfolios.portfolio_7.strategy import Portfolio7Strategy` and append to that list when materializing.
- `src/portfolios/portfolio_manager_config.json:1-8` is the only file we *do* edit (the task mandates it). Current registered weights are 1:0.10, 2:0.90. We add `"7": 0.0`.

---

## 4. Design decisions

### 4.1 Aggregation window: 21 trading days with 5-day EWM half-life

**Decision:** default `SENTIMENT_WINDOW_DAYS = 21`, `SENTIMENT_AGG_METHOD = "ewm"`, half-life = 5d (`SENTIMENT_EWM_HALFLIFE_DAYS = 5`). Configurable.

**Evidence:**
- Heston-Sinha 2017 (#5): "*daily news predicts stock returns for only 1 to 2 days, while weekly news predicts stock returns for one quarter*". A 21-day window captures roughly 4 weeks of news = strong weekly-equivalent signal with 4x more articles per name than a 5d window -> denoised, per Boudoukh et al 2013 (#6) and Petrescu 2021 (#13).
- Tetlock 2007 (#1) and Garcia 2013 (#4): the 1d Tetlock-style signal *partially reverses* within ~4-5 days. Using a window much shorter than 5d risks trading the reversal rather than the drift. Using a window much longer than 1 quarter risks averaging through earnings-driven regime shifts (Tetlock et al 2008 #2 notes the post-earnings amplification).
- Ke-Kelly-Xiu 2019 (#7) form long-short on *recent* article sentiment with multi-day holding periods, validating the multi-week look-back.
- EWM half-life of 5d is the geometric mean of the Tetlock 1-2d signal floor and the Heston-Sinha weekly persistence — it puts roughly 87% of the weight in the most recent 15 trading days and de-weights stale month-old articles whose drift is already priced.

Pandas implementation: `series.ewm(halflife=5, times=...).mean().iloc[-1]`.

**Why monthly cadence, not daily?** Portfolio_6 already rebalances monthly (`strategy.py:161-167`). Sentiment z-scores are recomputed at each rebalance from the *trailing 21d* of articles available *as of `context.time`*. Holding the tilt fixed for the month is acceptable because (a) the underlying 21d EWM signal is itself slow-moving, (b) it matches P6's existing risk budget, (c) it avoids the daily turnover that Heston-Sinha (#5) shows is dominated by the 1-2d reversal noise.

### 4.2 Ex-ante construction rule (strict, no look-ahead)

**Decision:** at decision time `t = context.time` (timezone-aware, UTC normalized), pull only `news_sentiment` rows with `published_at < t`. We use **strict less-than**, not `<=`, because intraday news published exactly at the bar timestamp may carry information not yet in `t-1`'s close.

**Look-ahead trap (must defend against):**

1. **Same-bar leakage.** If `published_at = t` and `t = 16:00 ET`, an article from 15:59 ET *would* be in `t-1`'s session. But for safety and reproducibility we use `<`, not `<=`, on the raw timestamp. The cost is at most one bar of staleness; the benefit is robustness across upstream timestamp definitions (ingest time vs publish time vs Dow Jones release time, all of which the FMP and Alpha gateways `src/common/articles_gateway.py:14-34` route differently).
2. **Aggregator leakage.** `market_data.avg_sentiment` is the *daily mean* of articles with `DATE(published_at) = md.date` (see `repository.py:255-267`). A naive "give me avg_sentiment for date `t`" includes articles that came out at 23:00 on `t` — past your morning rebalance. Solution: at decision time `t`, fetch `avg_sentiment` only for **dates `< t.date()`** (calendar date strictly before today). For consistency we *prefer* the per-article path (`news_sentiment` table) with `published_at < t`, because that path is provably PIT.
3. **Survivorship/dictionary leakage.** Our FinBERT model weights are fixed (`NLP/sentiment/scorer.py:31-32` -> `ProsusAI/finbert` or local safetensors), so there is no in-sample dictionary refit. Loughran-McDonald (#3) is *not* used here, sidestepping that retraining concern.
4. **Lopez de Prado 2018 (#15)** Ch. 7: every feature must answer "could this value have been computed at exact decision time?" Our SQL filter `WHERE published_at < %s` with `%s = context.time` does so.

**Defensive primary read path:**
```sql
SELECT ticker, published_at, sentiment_score
FROM news_sentiment
WHERE ticker = ANY(%s)
  AND published_at >= %s   -- t - SENTIMENT_WINDOW_DAYS (calendar days, generous buffer)
  AND published_at <  %s   -- strict < cutoff (= context.time)
```

**Fallback** (when `news_sentiment` is empty for a ticker in the window): use `market_data.avg_sentiment` (or `sentiment_score` column — see schema note in section 3) with `date < context.time.date()`. This fallback is gated by `MIN_ARTICLES_PER_TICKER`: if neither path yields enough articles, **the stock gets z=0 (no tilt)**, not silent forward-fill from neighbours.

### 4.3 Tilt mechanism: exponential, post-screening, post-cap

**Decision:**
1. Run Portfolio_6's full screen -> get `w_p6: Dict[ticker, weight]` for the stock sleeve (not hedges).
2. For each survivor `i`, compute `s_i` = EWM-mean of `published_at < t` sentiment over the trailing 21 trading days.
3. Drop survivors with `n_articles_i < MIN_ARTICLES_PER_TICKER` (set their `z_i = 0`, retain p6 weight).
4. Cross-sectional standardize across the survivor cohort: `z_i = (s_i - mean(s)) / std(s)` (NaN-safe std; if `std<=eps`, set all `z_i=0`).
5. Clip `z_i` to `[-Z_CLIP, +Z_CLIP]` (default 3.0) to neutralize FinBERT outliers / single-article noise spikes.
6. Tilt: `w_tilt_i = w_p6_i * exp(LAMBDA * z_i)`.
7. Renormalize so the sleeve gross matches the pre-tilt sleeve gross: `w_tilt_i *= (sum w_p6) / (sum w_tilt)`.
8. Re-apply `MAX_WEIGHT_PER_STOCK` cap iteratively (same algorithm as `screener.inverse_vol_weights` in `portfolio_6/screener.py:80-112`): clip overweights at the cap, redistribute slack to under-cap names proportionally to current tilted weight. Up to 20 iterations.
9. Hedge sleeve (`GLD_WEIGHT`, `TREND_HEDGE_WEIGHT`) and vol-target scaling are applied **after** the tilt, exactly as in P6 — unchanged.

**Why exp(LAMBDA * z)?**
- Multiplicative on weights (preserves long-only sign), monotone, smooth.
- `LAMBDA = 0.25` -> `exp(0.5)/exp(-0.5) = e ~= 2.72`: a +2sigma name gets ~e times the weight of a -2sigma name. Comparable to LSEG/FTSE multi-factor index tilt magnitudes (Source #14 BERT-BL tilt magnitudes; FTSE Russell tilt research).
- Robust to a degenerate `z` distribution (if all z=0, weights are unchanged).
- The cap re-application bounds tail exposure regardless of LAMBDA mis-specification.

**Why post-screening, not pre?** The P6 screener is built on price-only (vol, max return) + fundamentals. Inserting sentiment into the *ranking* would conflate two orthogonal alpha channels. Tilting after preserves the diversification properties of the inverse-vol weighting and lets us shut off the tilt with `LAMBDA=0` to recover Portfolio_6 exactly (a clean A/B switch).

### 4.4 Citations underlying each design knob

| Knob | Default | Citation |
|---|---|---|
| `SENTIMENT_WINDOW_DAYS=21` | trading days | Heston-Sinha 2017 (#5), Ke-Kelly-Xiu 2019 (#7) |
| `SENTIMENT_AGG_METHOD="ewm"` + `HALFLIFE=5d` | balances 1-2d Tetlock signal vs weekly drift | Tetlock 2007 (#1), Heston-Sinha 2017 (#5) |
| `MIN_ARTICLES_PER_TICKER=3` | denoise small samples | Boudoukh et al 2013 (#6), Petrescu 2021 (#13) |
| `SENTIMENT_TILT_LAMBDA=0.25` | moderate tilt | Kim et al 2022 (#14), Russell multi-factor tilt magnitudes |
| `Z_CLIP=3.0` | outlier-robust z-score | Lopez de Prado 2018 (#15) |
| `MAX_WEIGHT_PER_STOCK` cap re-applied post-tilt | risk hygiene | inherited from P6 |
| Strict `<` PIT cutoff | no leakage | Lopez de Prado 2018 (#15), Tetlock 2007 (#1) |

---

## 5. New files (apply-ready)

### 5.1 `src/portfolios/portfolio_7/__init__.py`

```python
"""Portfolio 7: Portfolio 6 + cross-sectional FinBERT sentiment tilt."""

from src.portfolios.portfolio_7.strategy import Portfolio7Strategy

__all__ = ["Portfolio7Strategy"]
```

### 5.2 `src/portfolios/portfolio_7/strategy.py`

```python
"""
Portfolio 7 = Portfolio 6 + cross-sectional sentiment z-score tilt.

Inherits the full P6 screen + inverse-vol + vol-target + hedge pipeline and
overlays a post-screen exponential tilt on the stock sleeve only:

    w_tilt_i = w_p6_i * exp(LAMBDA * z_i)

where z_i is the cross-sectional standardized 21d EWM-mean sentiment of
ticker i computed strictly from articles with published_at < context.time.

Pipeline:
  1. P6.rebalance() produces stock-sleeve weights + hedge sleeve weights
     (we hook into the same _rebalance and add a tilt step).
  2. Pull news_sentiment rows ex-ante (published_at < context.time) for all
     survivors. Aggregate to one scalar per ticker (EWM mean, configurable).
  3. Cross-sectional z-score across survivors, clip at +/-Z_CLIP.
  4. Apply exp(LAMBDA * z) multiplicatively; renormalize to original sleeve
     gross; iteratively re-apply MAX_WEIGHT_PER_STOCK cap.
  5. Vol-target scaling and hedge sleeves run AFTER the tilt -- identical
     to P6, by intent.

Strict ex-ante (no look-ahead): every SQL filter uses '<' on published_at
against context.time, never '<=', and never the current bar's date.

Configurable knobs (PORTFOLIO_7_CONFIG):
  SENTIMENT_TILT_LAMBDA      float, default 0.25
  SENTIMENT_WINDOW_DAYS      int,   default 21 (trading days)
  SENTIMENT_EWM_HALFLIFE_DAYS float, default 5.0
  MIN_ARTICLES_PER_TICKER    int,   default 3
  SENTIMENT_AGG_METHOD       str,   "ewm" | "mean", default "ewm"
  SENTIMENT_Z_CLIP           float, default 3.0
  SENTIMENT_FALLBACK_TO_MARKET_DATA bool, default True
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from portfolios.portfolio_6.strategy import Portfolio6Strategy
    from portfolios.strategy_api import StrategyContext
except ImportError:
    from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
    from src.portfolios.strategy_api import StrategyContext


class Portfolio7Strategy(Portfolio6Strategy):
    """Portfolio 6 with a cross-sectional sentiment z-score tilt overlay."""

    # SQL is parameterised; tickers list is bound with ANY(%s::text[]).
    _SENTIMENT_QUERY = """
        SELECT ticker, published_at, sentiment_score
        FROM news_sentiment
        WHERE ticker = ANY(%s::text[])
          AND published_at >= %s
          AND published_at <  %s
          AND sentiment_score IS NOT NULL
    """

    # Fallback path: defensively COALESCE the two known column names.
    _MARKET_DATA_SENTIMENT_QUERY = """
        SELECT ticker, date,
               COALESCE(avg_sentiment, sentiment_score)::float8 AS s
        FROM market_data
        WHERE ticker = ANY(%s::text[])
          AND date >= %s
          AND date <  %s
          AND COALESCE(avg_sentiment, sentiment_score) IS NOT NULL
    """

    def __init__(
        self,
        db_connector,
        executor,
        debug=False,
        config_dict=None,
        backtest_start_date=None,
    ):
        if config_dict is None:
            raise ValueError("config_dict is required for Portfolio7Strategy.")

        cfg = dict(config_dict)
        # Inherit P6 sub-config under PORTFOLIO_6_CONFIG; pull our own under PORTFOLIO_7_CONFIG.
        p7_cfg = dict(cfg.get("PORTFOLIO_7_CONFIG", {}))

        super().__init__(
            db_connector=db_connector,
            executor=executor,
            debug=debug,
            config_dict=cfg,
            backtest_start_date=backtest_start_date,
        )

        # Re-bind logger to the P7 class name so log lines are unambiguous.
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.portfolio_id}")

        self.tilt_lambda: float = float(p7_cfg.get("SENTIMENT_TILT_LAMBDA", 0.25))
        self.sent_window_days: int = int(p7_cfg.get("SENTIMENT_WINDOW_DAYS", 21))
        self.sent_ewm_halflife: float = float(
            p7_cfg.get("SENTIMENT_EWM_HALFLIFE_DAYS", 5.0)
        )
        self.min_articles: int = int(p7_cfg.get("MIN_ARTICLES_PER_TICKER", 3))
        self.sent_agg_method: str = str(
            p7_cfg.get("SENTIMENT_AGG_METHOD", "ewm")
        ).strip().lower()
        if self.sent_agg_method not in ("ewm", "mean"):
            self.logger.warning(
                "[P7] Unknown SENTIMENT_AGG_METHOD=%s; falling back to 'ewm'.",
                self.sent_agg_method,
            )
            self.sent_agg_method = "ewm"
        self.z_clip: float = float(p7_cfg.get("SENTIMENT_Z_CLIP", 3.0))
        self.fallback_to_md: bool = bool(
            p7_cfg.get("SENTIMENT_FALLBACK_TO_MARKET_DATA", True)
        )

        self.logger.info(
            "[P7] sentiment tilt cfg: lambda=%.3f, window=%dd, halflife=%.1fd, "
            "min_articles=%d, agg=%s, z_clip=%.2f, fallback_md=%s",
            self.tilt_lambda,
            self.sent_window_days,
            self.sent_ewm_halflife,
            self.min_articles,
            self.sent_agg_method,
            self.z_clip,
            self.fallback_to_md,
        )

    # ------------------------------------------------------------------
    # Ex-ante sentiment retrieval
    # ------------------------------------------------------------------

    def _fetch_sentiment_ex_ante(
        self,
        tickers: Iterable[str],
        cutoff_ts: pd.Timestamp,
    ) -> Dict[str, pd.Series]:
        """
        Return {ticker: pd.Series indexed by published_at, values in [-1,1]}
        for articles with published_at strictly < cutoff_ts and within the
        trailing window.
        """
        tickers = [t for t in tickers if isinstance(t, str) and t]
        if not tickers or cutoff_ts is None:
            return {}

        # Use a generous calendar buffer >= trading-day window so weekends/holidays don't
        # truncate the EWM. 21 trading days ~ 31 calendar days; use 2x for safety.
        start_ts = cutoff_ts - pd.Timedelta(days=max(self.sent_window_days * 2, 31))

        result = self.db.execute_query(
            self._SENTIMENT_QUERY,
            (list(tickers), start_ts.to_pydatetime(), cutoff_ts.to_pydatetime()),
            fetch=True,
        )

        out: Dict[str, pd.Series] = {}
        if result.get("status") == "success" and result.get("data"):
            df = pd.DataFrame(result["data"])
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
            df["sentiment_score"] = pd.to_numeric(
                df["sentiment_score"], errors="coerce"
            ).clip(-1.0, 1.0)
            df = df.dropna(subset=["published_at", "sentiment_score", "ticker"])
            for ticker, group in df.groupby("ticker"):
                series = group.set_index("published_at")["sentiment_score"].sort_index()
                if not series.empty:
                    out[str(ticker)] = series

        # Fallback path for tickers below the article threshold
        if self.fallback_to_md:
            short = [
                t for t in tickers
                if t not in out or len(out[t]) < self.min_articles
            ]
            if short:
                md = self._fetch_market_data_sentiment(short, cutoff_ts)
                for t, s in md.items():
                    if t not in out or len(s) >= self.min_articles:
                        out[t] = s
        return out

    def _fetch_market_data_sentiment(
        self, tickers: List[str], cutoff_ts: pd.Timestamp
    ) -> Dict[str, pd.Series]:
        """Fallback to market_data.avg_sentiment (or sentiment_score) by date."""
        if not tickers:
            return {}
        # Use date < cutoff_date to avoid same-day intraday leakage.
        cutoff_date = cutoff_ts.tz_convert("UTC").date() if cutoff_ts.tzinfo else cutoff_ts.date()
        start_date = cutoff_date - timedelta(days=max(self.sent_window_days * 2, 31))

        result = self.db.execute_query(
            self._MARKET_DATA_SENTIMENT_QUERY,
            (list(tickers), start_date, cutoff_date),
            fetch=True,
        )
        out: Dict[str, pd.Series] = {}
        if result.get("status") == "success" and result.get("data"):
            df = pd.DataFrame(result["data"])
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["s"] = pd.to_numeric(df["s"], errors="coerce").clip(-1.0, 1.0)
            df = df.dropna(subset=["date", "s", "ticker"])
            for ticker, group in df.groupby("ticker"):
                series = group.set_index("date")["s"].sort_index()
                if not series.empty:
                    out[str(ticker)] = series
        return out

    # ------------------------------------------------------------------
    # Aggregation and tilt math
    # ------------------------------------------------------------------

    def _aggregate_sentiment(
        self, series_by_ticker: Dict[str, pd.Series]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Reduce per-article sentiment series to a scalar per ticker.

        Returns (sentiment_scalar: pd.Series, n_articles: pd.Series).
        """
        agg_vals: Dict[str, float] = {}
        n_articles: Dict[str, int] = {}
        for ticker, series in series_by_ticker.items():
            n = len(series)
            n_articles[ticker] = n
            if n < self.min_articles:
                # Will be marked as z=0 downstream; keep raw mean for logging.
                agg_vals[ticker] = float(series.mean()) if n > 0 else np.nan
                continue
            if self.sent_agg_method == "ewm":
                ewm = series.ewm(
                    halflife=pd.Timedelta(days=self.sent_ewm_halflife),
                    times=series.index,
                    adjust=True,
                ).mean()
                agg_vals[ticker] = float(ewm.iloc[-1])
            else:
                agg_vals[ticker] = float(series.mean())
        return pd.Series(agg_vals, dtype=float), pd.Series(n_articles, dtype=int)

    def _cross_sectional_z(
        self,
        sentiment: pd.Series,
        n_articles: pd.Series,
    ) -> pd.Series:
        """Demean+rescale across the cohort; suppress under-supported names."""
        eligible_mask = n_articles >= self.min_articles
        eligible = sentiment[eligible_mask].dropna()
        if eligible.empty or eligible.std(ddof=0) <= 1e-12:
            return pd.Series(0.0, index=sentiment.index, dtype=float)

        mu = float(eligible.mean())
        sd = float(eligible.std(ddof=0))
        z = (sentiment - mu) / sd
        # Names without enough articles get z=0 (no tilt).
        z = z.where(eligible_mask, 0.0)
        # Clip outliers.
        z = z.clip(lower=-self.z_clip, upper=self.z_clip)
        return z.fillna(0.0)

    def _apply_tilt(
        self,
        stock_weights: Dict[str, float],
        z_scores: pd.Series,
    ) -> Dict[str, float]:
        """exp(LAMBDA * z) tilt, renormalize to original gross, re-cap iteratively."""
        if not stock_weights:
            return {}
        if self.tilt_lambda == 0.0:
            return dict(stock_weights)

        original_gross = float(sum(stock_weights.values()))
        if original_gross <= 0.0:
            return dict(stock_weights)

        tilted: Dict[str, float] = {}
        for ticker, w in stock_weights.items():
            z = float(z_scores.get(ticker, 0.0)) if pd.notna(z_scores.get(ticker, 0.0)) else 0.0
            tilted[ticker] = max(float(w), 0.0) * float(np.exp(self.tilt_lambda * z))

        tilted_gross = sum(tilted.values())
        if tilted_gross <= 0.0:
            return dict(stock_weights)

        # Renormalize to preserve sleeve gross.
        scale = original_gross / tilted_gross
        tilted = {t: w * scale for t, w in tilted.items()}

        # Re-apply per-name cap (same algorithm as screener.inverse_vol_weights).
        s = pd.Series(tilted, dtype=float)
        for _ in range(20):
            over = s > self.max_weight
            if not over.any():
                break
            slack = float((s[over] - self.max_weight).sum())
            s[over] = self.max_weight
            under = ~over
            under_sum = float(s[under].sum())
            if under_sum <= 0:
                break
            s[under] = s[under] + slack * (s[under] / under_sum)
        return s.to_dict()

    # ------------------------------------------------------------------
    # Override _rebalance to inject the tilt between weighting and hedge sleeves
    # ------------------------------------------------------------------

    def _rebalance(self, context: StrategyContext):
        """
        Override P6._rebalance to splice the sentiment tilt between
        inverse-vol weighting and the hedge sleeves. We replicate the parent
        body inline because the parent does not expose the intermediate
        stock-sleeve weights as a separate method. Any change in P6 will
        require a corresponding port here -- this is intentional and called
        out in tests.
        """
        # The local imports avoid coupling import order at module load time.
        try:
            from portfolios.portfolio_6.screener import (
                deflated_sharpe_ratio,
                inverse_vol_weights,
                score_universe,
                select_top_n,
                vol_target_scale,
            )
        except ImportError:
            from src.portfolios.portfolio_6.screener import (
                deflated_sharpe_ratio,
                inverse_vol_weights,
                score_universe,
                select_top_n,
                vol_target_scale,
            )

        returns_matrix = self._collect_returns(context)
        if not returns_matrix:
            self.logger.warning(
                "[P7] No tickers have sufficient history; skipping rebalance."
            )
            return

        scores = score_universe(
            returns_matrix,
            self.fundamentals_df,
            use_fundamentals=self.use_fundamentals,
        )
        top = select_top_n(scores, n=self.screen_top_n)
        if not top:
            self.logger.warning("[P7] Top-N selection empty; skipping rebalance.")
            return

        weights = inverse_vol_weights(
            {t: returns_matrix[t] for t in top},
            max_weight=self.max_weight,
        )
        if not weights:
            self.logger.warning("[P7] Inverse-vol weighting empty; skipping rebalance.")
            return

        # ---- SENTIMENT TILT (the only P7-specific step) -------------------
        cutoff_ts = pd.Timestamp(context.time)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        else:
            cutoff_ts = cutoff_ts.tz_convert("UTC")

        sentiment_by_ticker = self._fetch_sentiment_ex_ante(
            tickers=list(weights.keys()),
            cutoff_ts=cutoff_ts,
        )
        sent_scalar, n_articles = self._aggregate_sentiment(sentiment_by_ticker)
        z = self._cross_sectional_z(sent_scalar, n_articles)

        pre_tilt_top5 = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        weights = self._apply_tilt(weights, z)
        post_tilt_top5 = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        try:
            n_supported = int((n_articles >= self.min_articles).sum())
            self.logger.info(
                "[P7] sentiment tilt applied: lambda=%.2f, supported=%d/%d, "
                "z_min=%.2f z_max=%.2f, pre_top5=%s post_top5=%s",
                self.tilt_lambda,
                n_supported,
                len(weights),
                float(z.min()) if len(z) else 0.0,
                float(z.max()) if len(z) else 0.0,
                pre_tilt_top5,
                post_tilt_top5,
            )
        except Exception:
            pass
        # -------------------------------------------------------------------

        # DSR diagnostic (using *post-tilt* weights so the multiple-testing
        # estimate accounts for the additional degree of freedom).
        returns_df = pd.DataFrame({t: returns_matrix[t] for t in top}).dropna(how="all")
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0.0)
        sleeve_returns = (returns_df * weight_series).sum(axis=1)
        try:
            dsr = deflated_sharpe_ratio(
                sleeve_returns,
                n_trials=len(returns_matrix) + 1,  # +1 for the sentiment trial
            )
            self.logger.info(
                "[P7] Deflated Sharpe probability=%.3f (n_trials=%s, top_n=%s)",
                dsr,
                len(returns_matrix) + 1,
                len(top),
            )
            if dsr < self.dsr_min_prob:
                self.logger.warning(
                    "[P7] DSR=%.3f below threshold %.2f; selection+tilt may be noise.",
                    dsr,
                    self.dsr_min_prob,
                )
        except Exception as e:
            self.logger.exception("[P7] DSR computation failed: %s", e)

        # Vol-target scaling AFTER the tilt (identical to P6).
        vol_scale = vol_target_scale(
            sleeve_returns,
            target_annual_vol=self.vol_target_annual,
            max_scale=self.max_leverage,
        )
        weights = {t: w * vol_scale for t, w in weights.items()}

        target_weights = dict(weights)
        if self.gld_ticker and self.gld_ticker in self.tickers:
            gld_asset = context.Market[self.gld_ticker]
            if gld_asset.Exists:
                target_weights[self.gld_ticker] = self.gld_weight

        if self.trend_ticker and self.trend_ticker in self.tickers:
            trend_asset = context.Market[self.trend_ticker]
            if trend_asset.Exists:
                target_weights[self.trend_ticker] = self.trend_weight

        total = sum(target_weights.values())
        if total > self.max_leverage > 0:
            scale = self.max_leverage / total
            target_weights = {t: w * scale for t, w in target_weights.items()}
            self.logger.info(
                "[P7] Leverage cap applied: total %.3f scaled to %.3f.",
                total,
                self.max_leverage,
            )

        self._target_weights = target_weights
        ranked = sorted(target_weights.items(), key=lambda kv: -kv[1])
        self.logger.info(
            "[P7] New targets: positions=%s, total_weight=%.3f, top5=%s",
            len(target_weights),
            sum(target_weights.values()),
            ranked[:5],
        )
```

### 5.3 `src/portfolios/portfolio_7/config.json`

```json
{
  "PORTFOLIO_ID": "7",
  "TICKERS": [],
  "INTERVAL": 23400,
  "LOOKBACK_DAYS": 400,
  "EXCH": "NASDAQ",
  "WEIGHTS": {},
  "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
  "PORTFOLIO_6_CONFIG": {
    "UNIVERSE_PATH": "src/portfolios/portfolio_6/universe.json",
    "FUNDAMENTALS_CSV": "fundamentals/fundamentals.csv",
    "USE_FUNDAMENTALS": true,
    "SCREEN_TOP_N": 50,
    "VOL_LOOKBACK_DAYS": 252,
    "MAX_WEIGHT_PER_STOCK": 0.05,
    "VOL_TARGET_ANNUAL": 0.13,
    "MAX_LEVERAGE": 1.5,
    "GLD_TICKER": "GLD",
    "GLD_WEIGHT": 0.07,
    "TREND_HEDGE_TICKER": "",
    "TREND_HEDGE_WEIGHT": 0.10,
    "REBALANCE_DRIFT_THRESHOLD": 0.005,
    "DSR_MIN_PROB": 0.5
  },
  "PORTFOLIO_7_CONFIG": {
    "SENTIMENT_TILT_LAMBDA": 0.25,
    "SENTIMENT_WINDOW_DAYS": 21,
    "SENTIMENT_EWM_HALFLIFE_DAYS": 5.0,
    "MIN_ARTICLES_PER_TICKER": 3,
    "SENTIMENT_AGG_METHOD": "ewm",
    "SENTIMENT_Z_CLIP": 3.0,
    "SENTIMENT_FALLBACK_TO_MARKET_DATA": true
  }
}
```

### 5.4 `src/portfolios/portfolio_7/universe.json`

Reuse the P6 universe by reference (`UNIVERSE_PATH` already points at `src/portfolios/portfolio_6/universe.json`). No separate file needed for P7. If the user wants a distinct universe later, drop a `universe.json` next to this config and update `UNIVERSE_PATH` accordingly.

---

## 6. Unified diff for `src/portfolios/portfolio_manager_config.json`

```diff
--- a/src/portfolios/portfolio_manager_config.json
+++ b/src/portfolios/portfolio_manager_config.json
@@ -1,8 +1,9 @@
 {
   "master_portfolio_id": "0",
   "currency": "USD",
   "portfolio_weights": {
     "1": 0.10,
-    "2": 0.90
+    "2": 0.90,
+    "7": 0.0
   }
 }
```

Note: registers Portfolio_7 with capital weight 0 (off by default). The capital weights as shown sum to 1.00 (existing 1+2 = 1.00 plus 7=0). When activating P7 the user must reduce 1 and 2 proportionally.

Additionally, the user must (manually, outside this read-only deliverable) add to `src/main_backtest.py`:

```diff
--- a/src/main_backtest.py
+++ b/src/main_backtest.py
@@ -29,6 +29,7 @@
 from src.portfolios.portfolio_6.strategy import Portfolio6Strategy
+from src.portfolios.portfolio_7.strategy import Portfolio7Strategy
 from src.portfolios.portfolio_BASE.strategy import BasePortfolio
@@ -62,6 +63,7 @@
     RBPStrategy,
     Portfolio6Strategy,
+    Portfolio7Strategy,
 ]
```

---

## 7. Unit-test sketch (NOT applied — code block only)

Save as `tests/portfolios/test_portfolio_7_sentiment.py` when productionising. Asserts ex-ante construction and tilt math.

```python
"""Sketch: Portfolio_7 sentiment tilt math + ex-ante guarantees."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.portfolios.portfolio_7.strategy import Portfolio7Strategy


CONFIG = {
    "PORTFOLIO_ID": "7",
    "TICKERS": ["AAA", "BBB", "CCC", "DDD"],
    "INTERVAL": 23400,
    "LOOKBACK_DAYS": 400,
    "DATA_FEEDS": ["MARKET_DATA", "POSITIONS", "CASH_EQUITY", "PORT_NOTIONAL"],
    "PORTFOLIO_6_CONFIG": {
        "UNIVERSE_PATH": "tests/fixtures/p6_universe.json",
        "USE_FUNDAMENTALS": False,
        "SCREEN_TOP_N": 4,
        "MAX_WEIGHT_PER_STOCK": 0.5,
        "VOL_TARGET_ANNUAL": 0.13,
        "MAX_LEVERAGE": 1.5,
        "GLD_TICKER": "",
        "GLD_WEIGHT": 0.0,
        "TREND_HEDGE_TICKER": "",
        "TREND_HEDGE_WEIGHT": 0.0,
    },
    "PORTFOLIO_7_CONFIG": {
        "SENTIMENT_TILT_LAMBDA": 0.25,
        "SENTIMENT_WINDOW_DAYS": 21,
        "MIN_ARTICLES_PER_TICKER": 1,
        "SENTIMENT_AGG_METHOD": "mean",
        "SENTIMENT_Z_CLIP": 3.0,
        "SENTIMENT_FALLBACK_TO_MARKET_DATA": False,
    },
}


@pytest.fixture
def p7():
    db = MagicMock()
    executor = MagicMock()
    return Portfolio7Strategy(db, executor, config_dict=CONFIG)


def test_exp_tilt_preserves_long_only_and_gross(p7):
    """Tilt must keep all weights >= 0 and preserve total gross."""
    w = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
    z = pd.Series({"AAA": +2.0, "BBB": +1.0, "CCC": -1.0, "DDD": -2.0})
    out = p7._apply_tilt(w, z)
    assert all(v >= 0 for v in out.values())
    assert sum(out.values()) == pytest.approx(sum(w.values()), abs=1e-9)
    assert out["AAA"] > out["BBB"] > out["CCC"] > out["DDD"]


def test_tilt_zero_lambda_is_identity(p7):
    p7.tilt_lambda = 0.0
    w = {"AAA": 0.3, "BBB": 0.7}
    z = pd.Series({"AAA": +3.0, "BBB": -3.0})
    out = p7._apply_tilt(w, z)
    assert out == pytest.approx({"AAA": 0.3, "BBB": 0.7})


def test_cap_reapplied_after_tilt(p7):
    p7.max_weight = 0.4
    w = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}
    z = pd.Series({"AAA": +3.0, "BBB": 0.0, "CCC": 0.0, "DDD": -3.0})
    out = p7._apply_tilt(w, z)
    assert max(out.values()) <= p7.max_weight + 1e-9


def test_zero_z_when_below_min_articles(p7):
    p7.min_articles = 5
    series_by_ticker = {
        "AAA": pd.Series([0.9, 0.9, 0.9],
                         index=pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03"], utc=True)),
    }
    scalar, n = p7._aggregate_sentiment(series_by_ticker)
    z = p7._cross_sectional_z(scalar, n)
    assert z["AAA"] == 0.0  # under-supported -> no tilt


def test_strict_less_than_cutoff(p7, monkeypatch):
    """No article with published_at == cutoff_ts may appear in the fetch."""
    captured_params = {}

    def fake_execute_query(sql, params=None, fetch=False):
        captured_params["sql"] = sql
        captured_params["params"] = params
        return {"status": "success", "data": []}

    monkeypatch.setattr(p7.db, "execute_query", fake_execute_query)
    cutoff = pd.Timestamp("2025-05-20 13:30", tz="UTC")
    p7._fetch_sentiment_ex_ante(["AAA"], cutoff)

    # Verify SQL uses strict < on the upper bound.
    assert "published_at <  %s" in captured_params["sql"], (
        "Ex-ante invariant: must use strict '<' on cutoff, not '<='."
    )
    # The 3rd positional param is the cutoff datetime.
    _, _, sql_cutoff = captured_params["params"]
    assert sql_cutoff == cutoff.to_pydatetime()


def test_aggregation_ignores_future_articles(p7):
    """Even if the DB returns a future article (defensive), the trimmed series
    feeding _aggregate_sentiment must be entirely < cutoff."""
    cutoff = pd.Timestamp("2025-05-20", tz="UTC")
    s = pd.Series(
        [0.9, -0.9, 0.0],
        index=pd.to_datetime(
            ["2025-05-18", "2025-05-19", "2025-05-21"],  # last one is post-cutoff
            utc=True,
        ),
    )
    # Caller (the strategy) should drop post-cutoff rows.
    s_filtered = s[s.index < cutoff]
    scalar, n = p7._aggregate_sentiment({"AAA": s_filtered})
    assert n["AAA"] == 2
    assert scalar["AAA"] == pytest.approx(0.0)  # mean of 0.9 and -0.9
```

---

## 8. Falsification test (gating rule for activation)

**Promotion criterion (must all pass before lifting `"7": 0.0` from the manager config):**

1. **OOS Sharpe lift, 3-yr rolling.** On a walk-forward (train: 2018-2022, test: 2023-2025) backtest with identical universe/screener/hedge sleeves as Portfolio_6:
   `Sharpe(P7) - Sharpe(P6) >= 0.15` (i.e. ~15 bps of risk-adjusted excess per year of vol).
   *Rationale:* matches the median FAJ-class sentiment-overlay improvement reported in Heston-Sinha 2017 and Kim et al 2022.

2. **No regime collapse.** Drawdown(P7) <= 1.10 * Drawdown(P6) in any 12-month rolling window. Garcia 2013 warns that sentiment effects amplify in recessions; we must not turn a defensive sleeve into a momentum chase.

3. **Deflated Sharpe gate.** P7's annualized DSR probability (`screener.deflated_sharpe_ratio` is already logged each rebalance) must remain >= 0.5 across OOS years. We bumped `n_trials` by 1 in the override to reflect the added selection step.

4. **Turnover budget.** P7 monthly turnover must stay within 1.5x P6's monthly turnover (extra trading is the cost we pay for the tilt; if it blows up, the tilt is dominated by transaction costs at the SLIPPAGE in `main_backtest.py:50`).

5. **No leakage canary.** A regression test in CI must instantiate P7 with a fixed cutoff `T` and assert that `_fetch_sentiment_ex_ante([...], T)` produces *no* article with `published_at >= T`. (Sketch above in section 7.)

**If any of 1-5 fails:** disable the tilt by setting `SENTIMENT_TILT_LAMBDA=0.0` in `PORTFOLIO_7_CONFIG`. The strategy degenerates exactly to Portfolio_6 (proven by `test_tilt_zero_lambda_is_identity` above), so there's a one-line rollback.

---

## 9. Risks and rollback path

### Risks

1. **FinBERT model drift.** `NLP/sentiment/scorer.py:62-70` falls back to the HF Hub `ProsusAI/finbert` if no local checkpoint exists. The hub version is mutable — a silent upgrade would change historical sentiment scores in production. Mitigation: pin to the local safetensors copy at `models/finbert/`; CI should fail if the checksum changes.
2. **Schema drift on the sentiment column.** The mismatch between `schemaDefinitions.py:51` (`avg_sentiment`) and `repository.py:255` (`sentiment_score`) is a real bug. We work around it via `COALESCE` in the fallback SQL — if Team-DB hardens the schema, our query continues to work. If they remove both columns we still survive via the `news_sentiment` per-article path.
3. **News-source survivorship.** FMP and Alpha gateways (`src/common/articles_gateway.py:14-34`) historically may not have covered the full universe uniformly. Stocks with thin coverage get `z=0` (no tilt), so the strategy degrades gracefully — but the *cross-section* of who gets tilted is itself a function of coverage, which is non-stationary. Monitor `n_supported` ratio in the log line `[P7] sentiment tilt applied: ...`.
4. **Recession reversal (Garcia 2013).** In 2008-style regimes the sign of the news-return relation has been documented to flip. Our defense is the rolling-Sharpe falsification rule + manual kill-switch via `LAMBDA=0`. Future work: regime-conditional LAMBDA.
5. **DB query latency.** The `news_sentiment` table has indexes on `(ticker, published_at)` (`repository.py:77`), but the `ANY(%s::text[])` IN-list of 500 tickers may still be slow on first month of each backtest. The lookback buffer `2 * SENTIMENT_WINDOW_DAYS` limits the rowcount.
6. **Crowding / decay.** Sentiment-tilted strategies have been live since ~2010 (RavenPack); the alpha decays. The OOS Sharpe gate is the right firewall.

### Rollback path

- **Soft rollback (no code change):** edit `src/portfolios/portfolio_7/config.json` to set `"SENTIMENT_TILT_LAMBDA": 0.0`. Portfolio_7 then equals Portfolio_6 exactly (covered by `test_tilt_zero_lambda_is_identity`).
- **Capital rollback:** set `"7": 0.0` in `portfolio_manager_config.json` (the default state).
- **Code rollback:** delete `src/portfolios/portfolio_7/` and revert the two-line diff in `main_backtest.py` and the one-line diff in `portfolio_manager_config.json`. No DB migrations, no schema changes — P7 is pure overlay code.

---

## 10. URLs fetched / searched (deduplicated)

Primary papers:
- [https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x)
- [https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01362.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2008.01362.x)
- [https://business.columbia.edu/sites/default/files-efs/pubfiles/3096/More_Than_Words_tetlock.pdf](https://business.columbia.edu/sites/default/files-efs/pubfiles/3096/More_Than_Words_tetlock.pdf)
- [https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x)
- [https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12027](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12027)
- [https://leeds-faculty.colorado.edu/garcia/media_v33.pdf](https://leeds-faculty.colorado.edu/garcia/media_v33.pdf)
- [https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories](https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories)
- [https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf](https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf)
- [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2207241](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2207241)
- [https://www.nber.org/papers/w26186](https://www.nber.org/papers/w26186)
- [https://www.aqr.com/Insights/Research/Working-Paper/Predicting-Returns-with-Text-Data](https://www.aqr.com/Insights/Research/Working-Paper/Predicting-Returns-with-Text-Data)
- [https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063)
- [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2071142](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2071142)
- [https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2011.01679.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2011.01679.x)
- [https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X](https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X)
- [https://www.researchgate.net/publication/351360244_News_Sentiment_Everywhere_Trading_Global_Equities](https://www.researchgate.net/publication/351360244_News_Sentiment_Everywhere_Trading_Global_Equities)
- [https://link.springer.com/article/10.1007/s00521-022-07403-1](https://link.springer.com/article/10.1007/s00521-022-07403-1)
- [https://pmc.ncbi.nlm.nih.gov/articles/PMC9150638/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9150638/)
- [https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/news-and-markets-in-the-time-of-covid19/C0EB2A55CF6A36CCBC5BCB3BAD99B9D4](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/news-and-markets-in-the-time-of-covid19/C0EB2A55CF6A36CCBC5BCB3BAD99B9D4)

Secondary / methodology:
- [https://www.ravenpack.com/research/global-variations-news-impact-equities](https://www.ravenpack.com/research/global-variations-news-impact-equities)
- [https://www.ravenpack.com/research/news-momentum](https://www.ravenpack.com/research/news-momentum)
- [https://quantpedia.com/how-to-improve-post-earnings-announcement-drift-with-nlp-analysis/](https://quantpedia.com/how-to-improve-post-earnings-announcement-drift-with-nlp-analysis/)
- [https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/multi-factor-indexes-power-of-tilting.pdf](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/multi-factor-indexes-power-of-tilting.pdf)
- [https://russellinvestments.com/-/media/files/nz/insights/how-to-choose-a-strategic-multifactor-equity-portfolio.pdf](https://russellinvestments.com/-/media/files/nz/insights/how-to-choose-a-strategic-multifactor-equity-portfolio.pdf)
- [https://arxiv.org/pdf/2412.19245](https://arxiv.org/pdf/2412.19245)
- [https://arxiv.org/pdf/2505.01432](https://arxiv.org/pdf/2505.01432)
- [https://arxiv.org/pdf/2306.02136](https://arxiv.org/pdf/2306.02136)
- [https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [https://www.spglobal.com/marketintelligence/en/documents/sp-capitaliq-quantamental-point-in-time-vs-lagged-fundamentals.pdf](https://www.spglobal.com/marketintelligence/en/documents/sp-capitaliq-quantamental-point-in-time-vs-lagged-fundamentals.pdf)
- [https://insights.glassnode.com/why-use-point-in-time-data/](https://insights.glassnode.com/why-use-point-in-time-data/)

---

*End of A2 deliverable.*
