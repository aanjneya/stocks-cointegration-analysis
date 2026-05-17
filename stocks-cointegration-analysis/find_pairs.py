from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd


def calculate_hurst(series: pd.Series) -> float:

    ts = np.asarray(series.dropna(), dtype=float)
    #Not enough data points
    if ts.size < 100:
        return np.nan

    max_lag = min(ts.size // 2, 128)
    #Not enough data intervals
    if max_lag < 5:
        return np.nan
    #Lags are time windows spaced logarithmically
    lags = np.unique(np.round(np.logspace(np.log10(2), np.log10(max_lag), 24)).astype(int))
    log_pairs: list[tuple[float, float]] = []

    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        #too small sample size for std
        if diff.size < 20:
            continue

        tau = float(np.std(diff, ddof=1))

        if np.isfinite(tau) and tau > 0:
            log_pairs.append((float(np.log(lag)), float(np.log(tau))))

    #not enough data
    if len(log_pairs) < 5:
        return np.nan

    x = np.array([a for a, _ in log_pairs], dtype=float)
    y = np.array([b for _, b in log_pairs], dtype=float)
    h, _ = np.polyfit(x, y, 1)

    return float(np.clip(h, 0.0, 1.0))


def calculate_half_life(series: pd.Series) -> tuple[float, float]:

    s = np.asarray(series.dropna(), dtype=float)
    #sample size too small
    if s.size < 50:
        return np.nan, np.nan

    ds = np.diff(s)
    lag = s[:-1]

    m = np.isfinite(ds) & np.isfinite(lag)
    ds = ds[m]
    lag = lag[m]

    #sample size too small
    if ds.size < 30:
        return np.nan, np.nan

    x_mat = np.column_stack([np.ones(len(lag)), lag])
    coef, _, _, _ = np.linalg.lstsq(x_mat, ds, rcond=None)
    beta = float(coef[1])
    #No mean life
    if beta >= 0:
        return np.nan, beta
    lam = -beta
    half_life = float(np.log(2.0) / lam)
    
    return half_life, beta


def estimate_spread_ols(df: pd.DataFrame, stock_a: str, stock_b: str) -> pd.Series:
    sub = df[[stock_a, stock_b]].dropna()
    if len(sub) < 2:
        return pd.Series(dtype=float)
    y = sub[stock_a].to_numpy(dtype=float)
    x_b = sub[stock_b].to_numpy(dtype=float)
    x_mat = np.column_stack([np.ones(len(sub)), x_b])
    coef, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    spread = y - (coef[0] + coef[1] * x_b)
    return pd.Series(spread, index=sub.index)


def get_cointegrated_pairs(df, df_tickers):
    return FindPairs(df, df_tickers).get_pairs()


class FindPairs:
    def __init__(self, df, dfTickers):
        self.df = df
        self.dfTickers = dfTickers
        self.sectored_map = {}
        self.pairs = []

    def get_pairs(self) -> list[str]:
        for index, row in self.dfTickers.iterrows():
            ticker = row['Symbol']
            sector = row['GICS Sector']
            subsector = row['GICS Sub-Industry']

            if sector not in self.sectored_map:
                self.sectored_map[sector] = {}
            if subsector not in self.sectored_map[sector]:
                self.sectored_map[sector][subsector] = []

            self.sectored_map[sector][subsector].append(ticker)

        for i in self.sectored_map.values():
            for substocks in i.values():
                if len(substocks) > 1:
                    for j in range(len(substocks)):
                        stock1_name = substocks[j]
                        if stock1_name not in self.df.columns:
                            continue

                        for k in range(j + 1, len(substocks)):
                            stock2_name = substocks[k]
                            if stock2_name not in self.df.columns:
                                continue

                            pair_df = self.df[[stock1_name, stock2_name]].dropna()

                            if len(pair_df) > 100:
                                pval = coint(pair_df.iloc[:, 0], pair_df.iloc[:, 1])[1]
                                if pval < 0.05:
                                    self.pairs.append({'stockA': stock1_name, 'stockB': stock2_name, 'pValue': pval})

        self.pairs.sort(key=lambda p: p['pValue'])
        return self.pairs