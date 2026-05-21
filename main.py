from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from find_pairs import half_life, hurst, estimate_spread_ols, get_cointegrated_pairs
from get_data import FetchStocks
from ols import Strategy

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "stocks_sp500.csv"
OOS_START = "2025-01-01"

GLOBAL_PAIRS_CACHE = {}

DEFAULT_PARAMS = {
    'rebalance_step': 30,
    'lookback': 504,
    'entry_z1': 2.9,
    'entry_z2': 4.0,
    'stop_z': 5.1,
    'exit_z': 0.8,
    't1_weight': 0.6,
    'start_date': OOS_START,
    'end_date': "2026-09-30"
}

MIN_OVERLAP = 180
HURST_MAX = 0.45


def _build_rebalance_dates(oos_index, first, days_step):

    oos_sorted = oos_index.sort_values()
    subset = oos_sorted[oos_sorted >= first]

    if len(subset) == 0:
        return []

    dates = [pd.Timestamp(subset[0])]

    while True:
        cal_next = dates[-1] + pd.Timedelta(days=days_step)
        nxt = oos_sorted[oos_sorted >= cal_next]

        if len(nxt) == 0:
            break

        t_new = pd.Timestamp(nxt[0])
        if t_new == dates[-1]:
            break

        dates.append(t_new)

    return dates

def _resolve_col(sym, columns):

    s = str(sym)
    if s in columns:
        return s

    alt = s.replace(".", "-")
    if alt in columns:
        return alt

    return None

def run_walk_forward(params= None, df_master= None, df_tickers= None, verbose= False):

    if params is None:
        params = DEFAULT_PARAMS

    rebalance_step = params["rebalance_step"]
    lookback = params["lookback"]
    start_date = params.get("start_date", OOS_START)
    end_date = params.get("end_date", "2026-12-31")

    strat_risk_kw = {
        "entry_z1": params["entry_z1"],
        "entry_z2": params["entry_z2"],
        "stop_z": params["stop_z"],
        "exit_z": params["exit_z"],
        "t1_weight": params["t1_weight"]
    }

    if df_master is None or df_tickers is None:
        df_master = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        df_tickers = FetchStocks(str(CSV_PATH)).get_all_tickers()

    idx = df_master.index
    oos_ix = idx[(idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))]

    rebalance_dates = _build_rebalance_dates(oos_ix, pd.Timestamp(start_date), rebalance_step)
    if not rebalance_dates:
        return 0.0, 0.0

    daily_portfolio_return = pd.Series(0.0, index=oos_ix)
    log_rows =[]

    for seg_i, T_start in enumerate(rebalance_dates):
        if seg_i + 1 < len(rebalance_dates):
            T_next = rebalance_dates[seg_i + 1]
            seg_dates = oos_ix[(oos_ix >= T_start) & (oos_ix < T_next)]
        else:
            seg_dates = oos_ix[oos_ix >= T_start]

        if len(seg_dates) == 0:
            continue

        hist_upto = df_master.loc[:T_start]
        if len(hist_upto) < lookback:
            continue

        rank_df = hist_upto.tail(lookback)
        if len(rank_df) < MIN_OVERLAP:
            continue

        cache_key = f"{T_start.strftime('%Y-%m-%d')}_{lookback}"
        if cache_key not in GLOBAL_PAIRS_CACHE:
            GLOBAL_PAIRS_CACHE[cache_key] = get_cointegrated_pairs(rank_df, df_tickers)

        pairs = GLOBAL_PAIRS_CACHE[cache_key]

        train_start = pd.Timestamp(rank_df.index[0]).strftime("%Y-%m-%d")

        candidates = []
        for p in pairs:
            a, b = p["stockA"], p["stockB"]
            try:
                ca = _resolve_col(a, rank_df.columns)
                cb = _resolve_col(b, rank_df.columns)
                if not ca or not cb:
                    continue
                if rank_df[[ca, cb]].dropna().shape[0] < MIN_OVERLAP:
                    continue

                spread = estimate_spread_ols(rank_df, ca, cb)
                if spread.empty or len(spread) < MIN_OVERLAP:
                    continue

                h = hurst(spread)
                hl, beta_slope = half_life(spread)

                if np.isnan(h) or h >= HURST_MAX:
                    continue
                if np.isnan(hl) or beta_slope >= 0:
                    continue

                candidates.append({
                    "Stock_A": ca, "Stock_B": cb, "pValue": p["pValue"],
                    "hurst": h, "half_life": hl, "ou_lambda": -float(beta_slope),
                })
            except Exception:
                continue

        candidates.sort(key=lambda r: r["half_life"])
        top20 = []
        ticker_counts = {}

        for c in candidates:
            a, b = c["Stock_A"], c["Stock_B"]

            if ticker_counts.get(a, 0) >= 1 or ticker_counts.get(b, 0) >= 1:
                continue

            try:
                strat_is = Strategy(rank_df, a, b, half_life=c["half_life"], **strat_risk_kw)
                _, summ_is = strat_is.backtest(start_date=train_start)

                if summ_is["n_trades"] <= 0:
                    continue

                top20.append({**c, **summ_is})
                ticker_counts[a] = ticker_counts.get(a, 0) + 1
                ticker_counts[b] = ticker_counts.get(b, 0) + 1

                if len(top20) >= 20: break
            except Exception:
                continue

        if not top20:
            continue

        seg_portfolio = {}
        T_start_str = T_start.strftime("%Y-%m-%d")
        seg_end_str = seg_dates[-1].strftime("%Y-%m-%d")

        for row in top20:
            a, b = row["Stock_A"], row["Stock_B"]
            try:
                ma = _resolve_col(a, df_master.columns)
                mb = _resolve_col(b, df_master.columns)
                if not ma or not mb:
                    continue
                if df_master[[ma, mb]].dropna().shape[0] < MIN_OVERLAP:
                    continue

                strat_oos = Strategy(df_master, ma, mb, half_life=row["half_life"], **strat_risk_kw)
                strat_oos.backtest(start_date=T_start_str, transaction_cost_bps=5.0)

                pnl_seg = strat_oos._pnl.reindex(seg_dates).fillna(0.0)
                seg_portfolio[f"{ma}_{mb}"] = pnl_seg

                log_rows.append({
                    "rebalance_date": T_start_str, "segment_end": seg_end_str,
                    "Stock_A": ma, "Stock_B": mb, "pValue": row["pValue"],
                    "hurst": row["hurst"], "half_life": row["half_life"],
                    "segment_pair_return": float((1.0 + pnl_seg).prod() - 1.0),
                })
            except Exception:
                continue

        if not seg_portfolio:
            continue

        seg_df = pd.DataFrame(seg_portfolio)
        seg_daily = seg_df.mean(axis=1)
        daily_portfolio_return.loc[seg_daily.index] = seg_daily

    if verbose:
        pd.DataFrame(log_rows).to_csv(ROOT / "oos_survivors.csv", index=False)

    portfolio_equity_curve = (1.0 + daily_portfolio_return).cumprod()

    if len(portfolio_equity_curve):
        vol = daily_portfolio_return.std()
        p_sharpe = float(daily_portfolio_return.mean() / vol * np.sqrt(252)) if vol and vol > 0 else 0.0
        p_dd = float((portfolio_equity_curve / portfolio_equity_curve.cummax() - 1.0).min())

        if verbose:
            tot_ret = float(portfolio_equity_curve.iloc[-1] - 1.0) * 100
            print("\n" + "="*40)
            print("WALK-FORWARD OOS RESULTS")
            print("="*40)
            print(f"Rebalance Events:       {len(rebalance_dates)}")
            print(f"Total Portfolio Return: {tot_ret:.2f}%")
            print(f"Annualized Sharpe:      {p_sharpe:.2f}")
            print(f"Max Drawdown:           {p_dd*100:.2f}%")
            print("="*40)

            fig, ax = plt.subplots(figsize=(10, 4))
            portfolio_equity_curve.plot(ax=ax, color="navy", linewidth=1.5)
            ax.set_title(f"Walk-Forward Portfolio Equity ({params['rebalance_step']} days rebalance)")
            ax.set_ylabel("Equity Multiplier")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

        return p_sharpe, p_dd

    return 0.0, 0.0


if __name__ == "__main__":
    run_walk_forward(DEFAULT_PARAMS, verbose=True)
