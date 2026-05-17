import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class Strategy:

    def __init__(self, df, stockA, stockB, *, half_life: float, t1_weight: float = 0.70, entry_z1, entry_z2, exit_z,
                 stop_z, rolling_hedge=True):
        self.df = df[[stockA, stockB]].dropna()
        self.stockA = stockA
        self.stockB = stockB
        self.half_life = half_life
        self.t1_weight = t1_weight
        self.z_window = int(max(half_life * 2, 10))
        self.hedge_window = int(max(half_life * 4, 30))
        self.entry_z1 = entry_z1
        self.entry_z2 = entry_z2
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.rolling_hedge = rolling_hedge

        self.alpha, self.beta = None, None
        self.spread, self.z_score, self.position = None, None, None
        self._oos_start, self._equity, self._pnl = None, None, None

    def _rolling_ols(self, A, B, window):
        Bm = B.rolling(window).mean()
        Am = A.rolling(window).mean()
        Bc = B - Bm
        Ac = A - Am
        num = (Bc * Ac).rolling(window).sum()
        den = (Bc ** 2).rolling(window).sum()
        beta = num / den.replace(0, np.nan)
        alpha = Am - beta * Bm
        return alpha, beta

    def OLS(self):
        A = self.df[self.stockA]
        B = self.df[self.stockB]

        if self.rolling_hedge:
            alpha, beta = self._rolling_ols(A, B, self.hedge_window)
        else:
            C = sm.add_constant(B)
            model = sm.OLS(A, C).fit()
            alpha = pd.Series(model.params.iloc[0], index=A.index)
            beta = pd.Series(model.params.iloc[1], index=A.index)

        self.alpha = alpha
        self.beta = beta
        self.spread = A - (alpha + beta * B)

        past = self.spread.shift(1)
        roll_mean = past.rolling(self.z_window).mean()
        roll_std = past.rolling(self.z_window).std()

        self.z_score = (self.spread - roll_mean) / roll_std.replace(0, np.nan)
        self._roll_std_z = roll_std
        self._roll_vol_10 = self.spread.rolling(10).std()

    def _tranche_targets(self, current_z: float, z_prev: float = None) -> tuple:

        if current_z > self.stop_z or current_z < -self.stop_z:
            return 0.0, 0.0
        elif current_z > self.entry_z2:
            return 0.0, -1.0
        elif current_z > self.entry_z1:
            return 0.0, -self.t1_weight
        elif current_z < -self.entry_z2:
            return 1.0, 0.0
        elif current_z < -self.entry_z1:
            return self.t1_weight, 0.0
        elif abs(current_z) < self.exit_z:
            return 0.0, 0.0
        else:
            return None, None

    def generate_signals(self, start_date="2021-01-01"):
        if self.z_score is None:
            self.OLS()
        self._oos_start = pd.Timestamp(start_date)

        z = self.z_score
        idx = z.index
        zvals = z.to_numpy()
        n = len(zvals)

        pos = np.zeros(n, dtype=np.float64)
        current = 0.0
        xz, sz = self.exit_z, self.stop_z

        for i in range(n):
            if idx[i] < self._oos_start:
                pos[i] = 0.0
                current = 0.0
                continue

            zi = zvals[i]
            if np.isnan(zi):
                pos[i] = current
                continue

            z_prev = zvals[i - 1] if i > 0 else np.nan

            if abs(zi) >= sz:
                current = 0.0
                pos[i] = current
                continue

            if current > 0 and zi >= -xz:
                current = 0.0
                pos[i] = current
                continue
            if current < 0 and zi <= xz:
                current = 0.0
                pos[i] = current
                continue

            long_t, short_t = self._tranche_targets(zi, z_prev)

            v10 = self._roll_vol_10.iloc[i]
            vz = self._roll_std_z.iloc[i]

            if not np.isnan(v10) and not np.isnan(vz) and vz > 0 and v10 > 2.0 * vz:
                long_t, short_t = 0.0, 0.0

            if long_t is None or short_t is None:
                pos[i] = current
                continue

            if current > 0:
                if long_t > current: current = long_t
            elif current < 0:
                if short_t < current: current = short_t
            else:
                if long_t > 0 and short_t < 0:
                    current = long_t if zi < 0 else short_t
                elif long_t > 0:
                    current = long_t
                elif short_t < 0:
                    current = short_t

            pos[i] = current

        self.position = pd.Series(pos, index=idx, name="position")

    def backtest(self, transaction_cost_bps=0.0, start_date="2021-01-01"):
        if self.position is None or self._oos_start != pd.Timestamp(start_date):
            self.generate_signals(start_date=start_date)

        oos = self.position.index >= pd.Timestamp(start_date)

        A = self.df[self.stockA]
        B = self.df[self.stockB]

        hedge = self.beta.shift(1)
        pos_lag = self.position.shift(1).fillna(0.0)

        diff_a = A.diff()
        diff_b = B.diff()
        dollar_pnl = diff_a - hedge * diff_b
        gross_exposure = A.shift(1) + hedge.abs() * B.shift(1)

        spread_ret = dollar_pnl / gross_exposure.replace(0, np.nan)
        pnl_gross = pos_lag * spread_ret

        turnover = self.position.diff().abs().fillna(0.0)
        leg_notional = (1.0 + self.beta.shift(1).abs()).fillna(1.0)
        cost = turnover * (transaction_cost_bps / 10000.0) * leg_notional if transaction_cost_bps else 0.0

        pnl = (pnl_gross - cost).fillna(0.0)
        pnl = pnl.where(oos, 0.0)

        self._pnl = pnl
        equity = (1.0 + pnl).cumprod()
        self._equity = equity

        pnl_oos = pnl.loc[oos]
        vol = pnl_oos.std()
        sharpe = (pnl_oos.mean() / vol * np.sqrt(252)) if vol and vol > 0 else np.nan

        eq_oos = equity.loc[oos]
        dd = (eq_oos / eq_oos.cummax() - 1.0).min() if len(eq_oos) else 0.0

        n_trades = int((self.position.diff().abs().fillna(0.0).loc[oos] > 0).sum())

        return equity, {
            "total_return": float(eq_oos.iloc[-1] - 1.0) if len(eq_oos) else 0.0,
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "n_trades": n_trades,
            "oos_start": str(pd.Timestamp(start_date).date()),
        }

    def plotOLS(self, start_date=None):
        if self.z_score is None:
            self.OLS()
        self.z_score.plot(figsize=(10, 4))
        for lvl, sty in [(self.entry_z1, "--"), (self.entry_z2, "--")]:
            plt.axhline(lvl, color="red", linestyle=sty, alpha=0.75)
            plt.axhline(-lvl, color="green", linestyle=sty, alpha=0.75)
        plt.axhline(self.stop_z, color="darkred", linestyle="-", linewidth=1.0, alpha=0.6)
        plt.axhline(-self.stop_z, color="darkgreen", linestyle="-", linewidth=1.0, alpha=0.6)
        plt.axhline(self.exit_z, color="red", linestyle=":", alpha=0.5)
        plt.axhline(-self.exit_z, color="green", linestyle=":", alpha=0.5)
        plt.axhline(0, color="black", linewidth=0.8)
        if start_date is not None:
            plt.axvline(pd.Timestamp(start_date), color="blue", linestyle="--", alpha=0.7, label="OOS start")
            plt.legend(loc="upper right")
        plt.title(f"z-score: {self.stockA} vs {self.stockB}")
        plt.tight_layout()
        plt.show()