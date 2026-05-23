"""
quant_trader.py
---------------
Empirically grounded algorithmic trading pipeline with Phase 1 & 2 AI.
Features: 
- Dynamic S&P 500 Liquidity Screener
- Hidden Markov Model (HMM) Regime Risk Management
- XGBoost Cross-Sectional Ranking
- Inverse Volatility Weighting
- Friction-Aware Limit Order Execution
"""

import os
import time
import json
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from hmmlearn.hmm import GaussianHMM
from dotenv import load_dotenv

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    # API
    alpaca_key: str = os.getenv("APCA_API_KEY_ID", "")
    alpaca_secret: str = os.getenv("APCA_API_SECRET_KEY", "")
    is_paper: bool = os.getenv("APCA_PAPER", "true").lower() == "true"
    
    # Dynamic Universe Setup
    universe_size: int = 50  # We will dynamically screen the top 50 S&P 500 stocks
    universe: tuple = ()     # Will be populated at runtime
    
    # Parameters
    top_n: int = 5           # Buy the top 5 out of the 50
    vol_lookback: int = 30
    
    # HMM Regime Exposure Multipliers
    exposure_bull: float = 1.0
    exposure_neutral: float = 0.5
    exposure_bear: float = 0.0
    
    # Execution Friction Controls
    min_weight_drift: float = 0.05
    limit_order_padding: float = 0.002

    # Grossman-Zhou Drawdown Controller
    drawdown_floor_alpha: float = 0.85   # 15% max DD from peak
    drawdown_leverage_k: float = 1.0
    gz_state_path: str = "gz_state.json"

    # Sector Concentration Cap
    max_names_per_sector: int = 2       # at most N picks from any one GICS sector
    sector_cap_enabled: bool = True

    # Intraday Circuit Breaker (executes at order-submission time, complements GZ)
    breaker_enabled: bool = True
    breaker_dd_from_hwm: float = 0.07   # if live DD from HWM exceeds 7%, refuse new buys
    breaker_dd_intraday: float = 0.03   # if today's session DD exceeds 3%, refuse new buys

    # HRP Covariance Estimator
    use_ledoit_wolf: bool = True        # shrink sample cov before HRP; A/B toggle

    # Allocator Selection (HRP vs Fractional Multi-Asset Kelly)
    allocator: str = "hrp"            # "hrp" or "kelly"
    kelly_fraction: float = 0.25      # Fractional Kelly multiplier (Quarter Kelly default)
    kelly_max_weight: float = 0.40    # Per-name cap to prevent absurd concentration
    kelly_lookback: int = 252

    # Almgren-Chriss style execution slicing for large orders
    slicing_enabled: bool = False          # Disabled by default for daily live trade
    slicing_threshold_pct: float = 0.05    # slice orders > 5% of equity
    slicing_num_children: int = 4
    slicing_window_minutes: int = 5
    slicing_convexity: float = 2.0         # 1.0 = linear/TWAP, higher = more front-loaded

# -----------------------------------------------------------------------------
# PIPELINE COMPONENTS
# -----------------------------------------------------------------------------

class DataFetcher:
    """Handles all external market data ingestion via vectorized calls."""
    
    @staticmethod
    def get_dynamic_universe(top_n: int = 50) -> Tuple[List[str], Dict[str, str]]:
        """Scrapes S&P 500 tickers and screens for highest Average Daily Dollar Volume.
        Also returns a {ticker: GICS sector} map for sector-concentration capping."""
        print("🔍 Scanning S&P 500 for the most liquid stocks...")
        try:
            # Scrape current S&P 500 components with a User-Agent to bypass Wikipedia blocks
            import io
            import requests
            header = {"User-Agent": "Mozilla/5.0"}
            html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=header).text
            # pandas 2.x requires a file-like object; passing a raw string makes lxml treat it as a filename and fail
            table = pd.read_html(io.StringIO(html))
            df = table[0]
            # yfinance uses '-' instead of '.' for dual-class shares (e.g., BRK.B -> BRK-B)
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            sector_map_full = dict(zip(
                df['Symbol'].str.replace('.', '-', regex=False),
                df['GICS Sector']
            ))

            # Vectorized fetch of 30 days of data for all 500 stocks
            data = yf.download(tickers, period="1mo", interval="1d", auto_adjust=True, progress=False)

            # Dollar Volume = Close Price * Volume
            dollar_vol = (data['Close'] * data['Volume']).mean()

            # Sort highest to lowest and take the top_n
            liquid_tickers = dollar_vol.dropna().sort_values(ascending=False).head(top_n).index.tolist()
            sector_map = {t: sector_map_full[t] for t in liquid_tickers if t in sector_map_full}

            print(f"✅ Universe built: {len(liquid_tickers)} ultra-liquid stocks identified "
                  f"across {len(set(sector_map.values()))} GICS sectors.")
            return liquid_tickers, sector_map

        except Exception as e:
            print(f"⚠️ Screener failed, falling back to safe list. Error: {e}")
            return (["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "LLY", "V"], {})

    @staticmethod
    def fetch_daily_ohlc(symbols: List[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
        """Fetches 3 years of OHLC data to ensure the HMM has enough history to
        learn regimes AND that the feature engineer can build overnight/intraday
        decompositions. Returns {"Open": DataFrame, "Close": DataFrame, "Volume": DataFrame}."""
        if "SPY" not in symbols:
            symbols = list(symbols) + ["SPY"]

        print(f"📥 Fetching {period} of OHLC price data for {len(symbols)} symbols...")
        df = yf.download(symbols, period=period, interval="1d", auto_adjust=True, progress=False)

        if df.empty or "Close" not in df.columns or "Open" not in df.columns:
            raise ValueError("Failed to fetch OHLC market data from yfinance.")

        closes = df["Close"]
        opens = df["Open"]
        volumes = df["Volume"] if "Volume" in df.columns else pd.DataFrame()
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        if isinstance(opens, pd.Series):
            opens = opens.to_frame()
        if isinstance(volumes, pd.Series):
            volumes = volumes.to_frame()

        # Apply dropna cleanup to both, then intersect surviving columns so the
        # two frames stay aligned ticker-for-ticker. Volume is best-effort and
        # not part of the intersection — production trading uses only Open/Close.
        closes = closes.dropna(axis=1, how="all")
        opens = opens.dropna(axis=1, how="all")
        common_cols = closes.columns.intersection(opens.columns)
        vol_out = volumes.reindex(columns=common_cols) if not volumes.empty else pd.DataFrame()
        return {"Open": opens[common_cols], "Close": closes[common_cols], "Volume": vol_out}

class MarketRegimeHMM:
    """
    Phase 2 AI Risk Manager: Hidden Markov Model.
    Classifies the market into Bull, Neutral, or Bear regimes using SPY returns and volatility.
    """
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.spy = prices["SPY"].dropna()

    def detect_regime(self) -> Tuple[str, float]:
        """Trains the HMM, maps the hidden states, and returns today's regime."""
        print("🕵️‍♂️ Training HMM for Market Regime Detection...")
        
        # Silence the harmless hmmlearn math warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        returns = self.spy.pct_change().dropna()
        volatility = returns.rolling(10).std().dropna()
        
        common_idx = returns.index.intersection(volatility.index)
        features = pd.DataFrame({
            'Return': returns[common_idx],
            'Volatility': volatility[common_idx]
        })
        
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, tol=0.01, random_state=42)
        hmm.fit(features)
        hidden_states = hmm.predict(features)
        
        features['State'] = hidden_states
        
        state_metrics = []
        for state in range(3):
            state_data = features[features['State'] == state]
            mean_ret = state_data['Return'].mean()
            mean_vol = state_data['Volatility'].mean()
            sharpe_proxy = mean_ret / (mean_vol + 1e-6)
            state_metrics.append({'State': state, 'Sharpe': sharpe_proxy})
            
        state_metrics = sorted(state_metrics, key=lambda x: x['Sharpe'])
        state_map = {
            state_metrics[0]['State']: "BEAR",   
            state_metrics[1]['State']: "NEUTRAL",
            state_metrics[2]['State']: "BULL"    
        }
        
        features['Regime'] = features['State'].map(state_map)
        current_state = hidden_states[-1]
        current_regime = state_map[current_state]

        # Persist for downstream consumers (dashboard timeline overlay, etc.)
        self.regime_history = features
        self.state_map = state_map

        print(f"[Regime Detected] The market is currently in a {current_regime} state.")
        return current_regime

class MLSignalGenerator:
    """Phase 1 AI Brain: XGBoost cross-sectional ranker."""

    # Cross-sectional MAD winsorization multiplier. Cells beyond
    # median ± MAD_WINSOR_K · MAD are clipped per-date. Set very high
    # (e.g. 1e6) to effectively disable winsorization.
    MAD_WINSOR_K: float = 5.0

    def __init__(self, config: StrategyConfig, prices: pd.DataFrame, opens: pd.DataFrame,
                 sector_map: Dict[str, str] = None):
        self.cfg = config
        self.prices = prices
        self.opens = opens
        self.sector_map = sector_map or {}

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(span=window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(span=window, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _apply_savgol(self, series: pd.Series, window: int = 11, polyorder: int = 3) -> pd.Series:
        """Causal EWMA smoother (halflife=5). Replaces the original
        Savitzky-Golay (1964) implementation, which used a centered window
        (window_length=11) and therefore included prices from up to 5 days in
        the future in every smoothed value. Paired with a 5-day forward-return
        target, that gave the model the answer: disabling Savgol dropped the
        out-of-sample 60-day Rank IC from 0.398 to 0.142.

        EWMA halflife=5 is mathematically causal (s_t = α·x_t + (1-α)·s_{t-1}),
        so no future prices can enter, and it recovers most of the legitimate
        noise-reduction benefit (walk-forward IC 0.028 → 0.045 vs no smoothing
        on the rescreened S&P 500 panel).

        The ``window`` and ``polyorder`` parameters are retained in the
        signature for backwards compatibility but are unused. Function name
        kept to minimize churn at call sites."""
        return series.ewm(halflife=5, adjust=False).mean()
        # --- Original Savitzky-Golay implementation kept commented for reference.
        # --- DO NOT re-enable without first replacing the forward-return target
        # --- with a causal variant — the centered window leaks the label.
        # s = series.dropna()
        # if len(s) < window:
        #     return series
        # try:
        #     smoothed = savgol_filter(s.values, window_length=window, polyorder=polyorder, mode="nearest")
        #     return pd.Series(smoothed, index=s.index).reindex(series.index)
        # except Exception as e:
        #     print(f"⚠️ Savitzky-Golay failed, returning raw series: {e}")
        #     return series

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = []
        tickers = [c for c in df.columns if c != "SPY"]
        sanity_logged = False
        for ticker in tickers:
            p_raw = df[ticker]
            p = self._apply_savgol(p_raw)
            ret_1d = p.pct_change(1)
            ret_5d = p.pct_change(5)
            ret_20d = p.pct_change(20)
            vol_20d = ret_1d.rolling(20).std()
            rsi_14 = self._compute_rsi(p, 14)
            ma_slope = (p.rolling(20).mean() / p.rolling(50).mean()) - 1.0
            # Target uses RAW close — smoothing the label would leak the future.
            fwd_ret_5d = p_raw.pct_change(5).shift(-5)

            # Overnight / intraday decomposition uses RAW opens & closes —
            # smoothing the open/close jump would destroy the gap signal.
            if ticker not in self.opens.columns:
                # Cannot build overnight features without opens; skip ticker.
                continue
            o_raw = self.opens[ticker]
            overnight_ret = (o_raw / p_raw.shift(1)) - 1.0
            intraday_ret = (p_raw / o_raw) - 1.0
            overnight_ret_5d = overnight_ret.rolling(5).mean()
            overnight_neg = overnight_ret.clip(upper=0.0)

            # HAR-RV cascade: weekly + monthly realized variance (rv_d ≈ vol_20d, dropped).
            sq_ret = ret_1d.pow(2)
            rv_w = sq_ret.rolling(5).sum()
            rv_m = sq_ret.rolling(22).sum()

            if not sanity_logged:
                raw_std = p_raw.pct_change(1).std()
                smooth_std = ret_1d.std()
                print(f"📉 Causal EWMA smoothing applied (halflife=5) | "
                      f"{ticker} ret_1d std: raw={raw_std:.5f} smoothed={smooth_std:.5f}")
                print(f"🌙 Overnight/intraday decomposition | "
                      f"{ticker} mean overnight_ret={overnight_ret.mean():.6f} "
                      f"mean intraday_ret={intraday_ret.mean():.6f}")
                sanity_logged = True

            ticker_df = pd.DataFrame({
                "ticker": ticker, "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_20d": ret_20d,
                "vol_20d": vol_20d, "rsi_14": rsi_14, "ma_slope": ma_slope,
                "overnight_ret": overnight_ret, "intraday_ret": intraday_ret,
                "overnight_ret_5d": overnight_ret_5d, "overnight_neg": overnight_neg,
                "rv_w": rv_w, "rv_m": rv_m,
                "target_fwd_ret": fwd_ret_5d
            })
            features.append(ticker_df)

        dataset = pd.concat(features)
        print("dataset.index sample:", dataset.index[:3], "name:", dataset.index.name)

        # Cross-sectional MAD winsorization: for each date, cap each feature at
        # median ± k·MAD. target_fwd_ret is deliberately excluded so the model
        # can still learn from realized tails. NaN bounds (mad==0) leave cells
        # untouched, which avoids squashing tied groups onto the median.
        exclude_cols = {"ticker", "target_fwd_ret"}
        winsor_cols = [c for c in dataset.columns if c not in exclude_cols]
        medians = dataset.groupby(level=0)[winsor_cols].transform("median")
        abs_dev = (dataset[winsor_cols] - medians).abs()
        mads = abs_dev.groupby(level=0).transform("median").replace(0, np.nan)
        lower = medians - self.MAD_WINSOR_K * mads
        upper = medians + self.MAD_WINSOR_K * mads

        latest = dataset.index.max()
        latest_data = dataset.loc[[latest], winsor_cols]
        latest_lower = lower.loc[[latest]]
        latest_upper = upper.loc[[latest]]
        total_cells = int(latest_data.notna().sum().sum())
        below = int((latest_data < latest_lower).sum().sum())
        above = int((latest_data > latest_upper).sum().sum())
        clipped_pct = (100.0 * (below + above) / total_cells) if total_cells > 0 else 0.0
        print(f"🪟 MAD winsorization (k={self.MAD_WINSOR_K}): clipped "
              f"{clipped_pct:.2f}% of feature cells on {pd.Timestamp(latest).date()}")

        dataset[winsor_cols] = dataset[winsor_cols].clip(lower=lower, upper=upper)
        dataset = dataset.dropna()
        dataset["target_rank"] = dataset.groupby(dataset.index)["target_fwd_ret"].rank(pct=True)
        return dataset

    def _compute_rolling_rank_ic(self, dataset: pd.DataFrame, feature_cols: List[str],
                                 window: int = 60) -> Tuple[float, float]:
        """Out-of-sample Rank IC diagnostic. Retrains on data strictly before the
        held-out window, predicts each held-out day, then computes cross-sectional
        Spearman(predictions, target_fwd_ret) per day. Returns (mean_IC, IC_IR)
        where IC_IR = mean(daily_ICs) / std(daily_ICs)."""
        unique_dates = sorted(dataset.index.unique())
        if len(unique_dates) < window + 10:
            return float("nan"), float("nan")

        held_out_dates = unique_dates[-window:]
        cutoff = held_out_dates[0]
        # Purge gap >= forward-return horizon: target_fwd_ret = close.shift(-5),
        # so the last 5 train rows' labels are realized inside the test window.
        purge_days = 5
        train = dataset[dataset.index < (cutoff - pd.Timedelta(days=purge_days))]
        test = dataset[dataset.index >= cutoff]
        if len(train) == 0 or len(test) == 0:
            return float("nan"), float("nan")

        diag_model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.05, random_state=42)
        diag_model.fit(train[feature_cols], train["target_rank"])
        test = test.copy()
        test["_pred"] = diag_model.predict(test[feature_cols])

        daily_ics = []
        for date in held_out_dates:
            day = test.loc[test.index == date]
            if len(day) < 5:
                continue
            if day["_pred"].nunique() < 2 or day["target_fwd_ret"].nunique() < 2:
                continue
            ic, _ = spearmanr(day["_pred"].values, day["target_fwd_ret"].values)
            if pd.notna(ic):
                daily_ics.append(float(ic))

        if not daily_ics:
            return float("nan"), float("nan")
        arr = np.array(daily_ics)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std())
        ic_ir = mean_ic / std_ic if std_ic > 0 else float("nan")
        return mean_ic, ic_ir

    def _compute_shuffled_ic(self, dataset: pd.DataFrame, feature_cols: List[str],
                             window: int = 60) -> Tuple[float, float]:
        """Definitive leakage sanity check. Same train/test split as
        _compute_rolling_rank_ic (with purge gap), but train labels are
        permuted before fitting. If features are leak-free, the model
        learns nothing and test-set Spearman IC ≈ 0. A materially nonzero
        IC here means there is a path from train features to test labels
        that survives label scrambling — i.e. structural leakage."""
        unique_dates = sorted(dataset.index.unique())
        if len(unique_dates) < window + 10:
            return float("nan"), float("nan")

        held_out_dates = unique_dates[-window:]
        cutoff = held_out_dates[0]
        purge_days = 5
        train = dataset[dataset.index < (cutoff - pd.Timedelta(days=purge_days))]
        test = dataset[dataset.index >= cutoff]
        if len(train) == 0 or len(test) == 0:
            return float("nan"), float("nan")

        rng = np.random.default_rng(42)
        train = train.copy()
        train["target_rank"] = rng.permutation(train["target_rank"].values)

        diag_model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.05, random_state=42)
        diag_model.fit(train[feature_cols], train["target_rank"])
        test = test.copy()
        test["_pred"] = diag_model.predict(test[feature_cols])

        daily_ics = []
        for date in held_out_dates:
            day = test.loc[test.index == date]
            if len(day) < 5:
                continue
            if day["_pred"].nunique() < 2 or day["target_fwd_ret"].nunique() < 2:
                continue
            ic, _ = spearmanr(day["_pred"].values, day["target_fwd_ret"].values)
            if pd.notna(ic):
                daily_ics.append(float(ic))

        if not daily_ics:
            return float("nan"), float("nan")
        arr = np.array(daily_ics)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std())
        ic_ir = mean_ic / std_ic if std_ic > 0 else float("nan")
        return mean_ic, ic_ir

    def _compute_walkforward_ic(self, dataset: pd.DataFrame, feature_cols: List[str],
                                train_window: int = 504, purge_days: int = 5,
                                test_window: int = 20, step: int = 20) -> Dict[str, float]:
        """Walk-forward Rank IC: rolling 504-day train, 5-day purge gap, 20-day
        non-overlapping test windows. Re-fits XGBoost per walk and concatenates
        per-day cross-sectional Spearman ICs across the full history. Far more
        robust than the single 60-day window: averages over many market regimes
        and gives an annualized IR. Returns a dict with n_days, mean_ic, std_ic,
        ic_ir_annualized, hit_rate."""
        unique_dates = sorted(dataset.index.unique())
        n = len(unique_dates)
        min_required = train_window + purge_days + test_window
        if n < min_required:
            print(f"⚠️ Walk-forward needs ≥{min_required} dates, have {n}. Skipping.")
            return {"n_days": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                    "ic_ir_ann": float("nan"), "hit_rate": float("nan"), "n_walks": 0}

        all_daily_ics: List[float] = []
        n_walks = 0
        test_start_idx = train_window + purge_days
        while test_start_idx + test_window <= n:
            test_dates = unique_dates[test_start_idx : test_start_idx + test_window]
            train_end_idx = test_start_idx - purge_days
            train_start_idx = train_end_idx - train_window
            train_dates = unique_dates[train_start_idx : train_end_idx]

            train_date_set = set(train_dates)
            test_date_set = set(test_dates)
            train = dataset.loc[dataset.index.isin(train_date_set)]
            test = dataset.loc[dataset.index.isin(test_date_set)]
            if len(train) == 0 or len(test) == 0:
                test_start_idx += step
                continue

            model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.05, random_state=42)
            model.fit(train[feature_cols], train["target_rank"])
            test = test.copy()
            test["_pred"] = model.predict(test[feature_cols])

            for date in test_dates:
                day = test.loc[test.index == date]
                if len(day) < 5:
                    continue
                if day["_pred"].nunique() < 2 or day["target_fwd_ret"].nunique() < 2:
                    continue
                ic, _ = spearmanr(day["_pred"].values, day["target_fwd_ret"].values)
                if pd.notna(ic):
                    all_daily_ics.append(float(ic))

            n_walks += 1
            test_start_idx += step

        if not all_daily_ics:
            return {"n_days": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                    "ic_ir_ann": float("nan"), "hit_rate": float("nan"), "n_walks": n_walks}

        arr = np.array(all_daily_ics)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        ic_ir_ann = mean_ic / std_ic * np.sqrt(252) if std_ic and std_ic > 0 else float("nan")
        hit_rate = float((arr > 0).mean())
        return {"n_days": len(arr), "mean_ic": mean_ic, "std_ic": std_ic,
                "ic_ir_ann": ic_ir_ann, "hit_rate": hit_rate, "n_walks": n_walks}

    def _compute_walkforward_ic_rescreened(self, closes: pd.DataFrame, dollar_vol: pd.DataFrame,
                                           feature_cols: List[str], top_n: int = 50,
                                           train_window: int = 504, purge_days: int = 5,
                                           test_window: int = 20, step: int = 20) -> Dict[str, float]:
        """Walk-forward IC with PER-WALK universe re-screening by training-window
        dollar volume. Removes the today's-liquidity-rank look-ahead from the
        standard walk-forward (which trains on a today-selected fixed panel).
        Features are recomputed per walk on the walk-specific universe so
        cross-sectional winsorization and target_rank use the same subset the
        model trains on."""
        unique_dates = sorted(closes.index.unique())
        n = len(unique_dates)
        min_required = train_window + purge_days + test_window
        if n < min_required:
            print(f"⚠️ Rescreened walk-forward needs ≥{min_required} dates, have {n}. Skipping.")
            return {"n_days": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                    "ic_ir_ann": float("nan"), "hit_rate": float("nan"), "n_walks": 0}

        all_daily_ics: List[float] = []
        n_walks = 0
        test_start_idx = train_window + purge_days
        universe_overlap_pct: List[float] = []
        last_walk_universe: set = set()
        while test_start_idx + test_window <= n:
            test_dates = unique_dates[test_start_idx : test_start_idx + test_window]
            train_end_idx = test_start_idx - purge_days
            train_start_idx = train_end_idx - train_window
            train_dates = unique_dates[train_start_idx : train_end_idx]

            # Re-screen: top-N by training-window mean dollar volume.
            adv = dollar_vol.loc[dollar_vol.index.isin(set(train_dates))].mean(skipna=True)
            walk_tickers = [t for t in adv.dropna().sort_values(ascending=False).head(top_n).index
                            if t != "SPY"]
            if len(walk_tickers) < 5:
                test_start_idx += step
                continue

            current_set = set(walk_tickers)
            if last_walk_universe:
                overlap = len(current_set & last_walk_universe) / len(current_set | last_walk_universe)
                universe_overlap_pct.append(overlap)
            last_walk_universe = current_set

            walk_closes = closes[walk_tickers].copy()
            try:
                walk_dataset = self._engineer_features(walk_closes)
            except Exception as e:
                print(f"⚠️ Walk feature engineering failed (skipping walk): {e}")
                test_start_idx += step
                continue

            train_date_set = set(train_dates)
            test_date_set = set(test_dates)
            train = walk_dataset.loc[walk_dataset.index.isin(train_date_set)]
            test = walk_dataset.loc[walk_dataset.index.isin(test_date_set)]
            if len(train) == 0 or len(test) == 0:
                test_start_idx += step
                continue

            model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.05, random_state=42)
            model.fit(train[feature_cols], train["target_rank"])
            test = test.copy()
            test["_pred"] = model.predict(test[feature_cols])

            for date in test_dates:
                day = test.loc[test.index == date]
                if len(day) < 5:
                    continue
                if day["_pred"].nunique() < 2 or day["target_fwd_ret"].nunique() < 2:
                    continue
                ic, _ = spearmanr(day["_pred"].values, day["target_fwd_ret"].values)
                if pd.notna(ic):
                    all_daily_ics.append(float(ic))

            n_walks += 1
            test_start_idx += step

        if not all_daily_ics:
            return {"n_days": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                    "ic_ir_ann": float("nan"), "hit_rate": float("nan"), "n_walks": n_walks}

        arr = np.array(all_daily_ics)
        mean_ic = float(arr.mean())
        std_ic = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        ic_ir_ann = mean_ic / std_ic * np.sqrt(252) if std_ic and std_ic > 0 else float("nan")
        hit_rate = float((arr > 0).mean())
        avg_overlap = float(np.mean(universe_overlap_pct)) if universe_overlap_pct else float("nan")
        return {"n_days": len(arr), "mean_ic": mean_ic, "std_ic": std_ic,
                "ic_ir_ann": ic_ir_ann, "hit_rate": hit_rate, "n_walks": n_walks,
                "avg_walk_to_walk_universe_overlap": avg_overlap}

    def run_cmda_diagnostic(self, dataset: pd.DataFrame,
                            dist_threshold: float = 0.5,
                            n_permutations: int = 10,
                            random_state: int = 42) -> pd.DataFrame:
        """Cluster-Based Mean Decrease Accuracy (López de Prado).

        Diagnostic-only: groups feature_cols into clusters by correlation-distance
        single-linkage hierarchical clustering, then measures the drop in
        out-of-sample Spearman IC when each cluster's features are jointly
        permuted on a held-out window. Joint permutation preserves within-cluster
        correlation so the importance reflects the cluster's unique contribution,
        not artifacts from breaking inter-feature dependence."""
        feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "ma_slope",
                        "overnight_ret", "intraday_ret", "overnight_ret_5d",
                        "overnight_neg", "rv_w", "rv_m"]

        corr = dataset[feature_cols].corr().clip(-1.0, 1.0)
        dist = np.sqrt(0.5 * (1.0 - corr))
        link = linkage(squareform(dist.values, checks=False), method="single")
        cluster_ids = fcluster(link, t=dist_threshold, criterion="distance")

        clusters: Dict[int, List[str]] = {}
        for feat, cid in zip(feature_cols, cluster_ids):
            clusters.setdefault(int(cid), []).append(feat)

        print(f"\n🔬 cMDA: {len(clusters)} cluster(s) at distance threshold {dist_threshold}")
        for cid, feats in sorted(clusters.items()):
            print(f"   Cluster {cid}: {feats}")

        unique_dates = sorted(dataset.index.unique())
        if len(unique_dates) < 10:
            raise ValueError("cMDA: insufficient unique dates for train/val split")
        split_idx = int(len(unique_dates) * 0.7)
        cutoff = unique_dates[split_idx]
        train = dataset[dataset.index < cutoff]
        val = dataset[dataset.index >= cutoff].copy()
        if len(train) == 0 or len(val) == 0:
            raise ValueError("cMDA: empty train or validation partition")
        print(f"   Train rows: {len(train)} (< {pd.Timestamp(cutoff).date()})  "
              f"Val rows: {len(val)} (≥ {pd.Timestamp(cutoff).date()})")

        model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                 learning_rate=0.05, random_state=random_state)
        model.fit(train[feature_cols], train["target_rank"])

        X_val = val[feature_cols].copy()
        y_val = val["target_rank"].values
        base_pred = model.predict(X_val)
        baseline_ic, _ = spearmanr(base_pred, y_val)
        if pd.isna(baseline_ic):
            baseline_ic = 0.0
        print(f"   Baseline Spearman IC (val, predictions vs target_rank): {baseline_ic:.4f}")

        rng = np.random.default_rng(random_state)
        n_val = len(X_val)
        results = []
        for cid, feats in sorted(clusters.items()):
            drops = np.empty(n_permutations)
            for i in range(n_permutations):
                perm_idx = rng.permutation(n_val)
                X_perm = X_val.copy()
                X_perm[feats] = X_val[feats].values[perm_idx]
                perm_pred = model.predict(X_perm)
                perm_ic, _ = spearmanr(perm_pred, y_val)
                if pd.isna(perm_ic):
                    perm_ic = 0.0
                drops[i] = baseline_ic - perm_ic
            results.append({
                "cluster_id": cid,
                "features_in_cluster": feats,
                "mean_importance": float(drops.mean()),
                "std_importance": float(drops.std()),
            })

        out = (pd.DataFrame(results)
               .sort_values("mean_importance", ascending=False)
               .reset_index(drop=True))
        out["rank"] = out.index + 1

        print("\n📋 cMDA Feature-Cluster Importance (higher = more useful to the model):")
        with pd.option_context("display.max_colwidth", None, "display.width", 200):
            print(out.to_string(index=False))
        return out

    def calculate_ml_scores(self) -> pd.Series:
        print("🤖 Training XGBoost on dynamic universe...")
        dataset = self._engineer_features(self.prices)
        latest_date = dataset.index.max()
        train_data = dataset[dataset.index < (latest_date - pd.Timedelta(days=5))]
        feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "ma_slope",
                        "overnight_ret", "intraday_ret", "overnight_ret_5d",
                        "overnight_neg", "rv_w", "rv_m"]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(train_data[feature_cols], train_data["target_rank"])

        mean_ic, ic_ir = self._compute_rolling_rank_ic(dataset, feature_cols, window=60)
        print(f"📊 Out-of-sample Rank IC (60d): {mean_ic:.3f}  |  IC IR: {ic_ir:.2f}")
        self.last_ic_metrics = {"mean_ic": mean_ic, "ic_ir": ic_ir, "window": 60}

        shuf_ic, shuf_ir = self._compute_shuffled_ic(dataset, feature_cols, window=60)
        print(f"🧪 Shuffled-target IC (should be ~0): {shuf_ic:.4f}  IR: {shuf_ir:.2f}")
        self.last_ic_metrics["shuffled_ic"] = shuf_ic
        self.last_ic_metrics["shuffled_ir"] = shuf_ir

        wf = self._compute_walkforward_ic(dataset, feature_cols)
        print(f"🚶 Walk-forward IC | walks={wf['n_walks']} test_days={wf['n_days']} "
              f"mean={wf['mean_ic']:.4f} std={wf['std_ic']:.4f} "
              f"IR(ann)={wf['ic_ir_ann']:.2f} hit_rate={wf['hit_rate']:.2%}")
        self.last_ic_metrics["walkforward"] = wf

        todays_data = []
        tickers = [c for c in self.prices.columns if c != "SPY"]
        for ticker in tickers:
            p_raw = self.prices[ticker].dropna()
            if len(p_raw) < 50: continue
            if ticker not in self.opens.columns:
                continue
            o_raw = self.opens[ticker].dropna()
            if len(o_raw) < 25:
                continue
            # Smooth then read ONLY the rightmost value, matching training-time alignment.
            p = self._apply_savgol(p_raw)
            try:
                ret_1d_series = p.pct_change(1)
                overnight_ret_series = (o_raw / p_raw.shift(1)) - 1.0
                intraday_ret_series = (p_raw / o_raw) - 1.0
                sq_ret_series = ret_1d_series.pow(2)

                overnight_ret_val = overnight_ret_series.iloc[-1]
                intraday_ret_val = intraday_ret_series.iloc[-1]
                overnight_ret_5d_val = overnight_ret_series.rolling(5).mean().iloc[-1]
                overnight_neg_val = min(overnight_ret_val, 0.0) if pd.notna(overnight_ret_val) else np.nan
                rv_w_val = sq_ret_series.rolling(5).sum().iloc[-1]
                rv_m_val = sq_ret_series.rolling(22).sum().iloc[-1]

                # Skip ticker entirely if any new feature is NaN — better than
                # poisoning the prediction frame.
                if any(pd.isna(v) for v in [overnight_ret_val, intraday_ret_val,
                                            overnight_ret_5d_val, overnight_neg_val,
                                            rv_w_val, rv_m_val]):
                    continue
            except Exception:
                continue

            todays_features = {
                "ticker": ticker, "ret_1d": p.pct_change(1).iloc[-1], "ret_5d": p.pct_change(5).iloc[-1],
                "ret_20d": p.pct_change(20).iloc[-1], "vol_20d": p.pct_change(1).rolling(20).std().iloc[-1],
                "rsi_14": self._compute_rsi(p, 14).iloc[-1],
                "ma_slope": (p.rolling(20).mean().iloc[-1] / p.rolling(50).mean().iloc[-1]) - 1.0,
                "overnight_ret": overnight_ret_val,
                "intraday_ret": intraday_ret_val,
                "overnight_ret_5d": overnight_ret_5d_val,
                "overnight_neg": overnight_neg_val,
                "rv_w": rv_w_val,
                "rv_m": rv_m_val,
            }
            todays_data.append(todays_features)

        today_df = pd.DataFrame(todays_data).set_index("ticker")
        # XGBoost returns float32; widen to float64 so downstream residual
        # writes don't trip pandas' strict dtype guard.
        today_df["predicted_rank"] = model.predict(today_df[feature_cols]).astype(np.float64)

        # Sector-neutralize the signal at PREDICTION time only — training still
        # targets raw cross-sectional rank. Regress predicted_rank on sector
        # dummies and replace with residuals so the surviving signal is the
        # within-sector component (skip if no sector_map or only one sector).
        if self.sector_map:
            sectors_today = pd.Series(
                {t: self.sector_map.get(t) for t in today_df.index}
            ).dropna()
            n_sectors = int(sectors_today.nunique())
            if len(sectors_today) > 0 and n_sectors > 1:
                valid_tickers = sectors_today.index.tolist()
                dummies = pd.get_dummies(sectors_today).astype(float)
                y = today_df.loc[valid_tickers, "predicted_rank"].values.astype(np.float64)
                X = dummies.values
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    residuals = y - X @ beta
                    today_df.loc[valid_tickers, "predicted_rank"] = residuals
                    print(f"🧹 Sector-neutralized signal: residualized across {n_sectors} sectors")
                except np.linalg.LinAlgError as e:
                    print(f"⚠️ Sector residualization failed ({e}); using raw predictions.")

        return today_df["predicted_rank"]

class PortfolioOptimizer:
    """Risk allocation via Hierarchical Risk Parity (López de Prado 2016).
    Clusters correlated names, quasi-diagonalizes the covariance, and recursively
    bisects weight by inverse cluster variance — prevents the hidden single-factor
    concentration that inverse-vol produces on a mega-cap-dominated universe."""

    def __init__(self, config: StrategyConfig, prices: pd.DataFrame):
        self.cfg = config
        self.prices = prices

    def _calculate_inverse_vol_weights(self, selected_tickers: List[str]) -> pd.Series:
        """Inverse-volatility fallback (kept private for use when HRP cannot run)."""
        recent_prices = self.prices[selected_tickers].tail(self.cfg.vol_lookback)
        volatility = recent_prices.pct_change().dropna().std().replace(0, 1e-6)
        inv_vol = 1.0 / volatility
        return inv_vol / inv_vol.sum()

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
        sub = cov.loc[items, items].values
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    def _recursive_bisection(self, cov: pd.DataFrame, ordered: List[str]) -> pd.Series:
        w = pd.Series(1.0, index=ordered)
        clusters = [ordered]
        while clusters:
            new_clusters = []
            for c in clusters:
                if len(c) <= 1:
                    continue
                mid = len(c) // 2
                left, right = c[:mid], c[mid:]
                var_l = self._cluster_var(cov, left)
                var_r = self._cluster_var(cov, right)
                alpha = 1.0 - var_l / (var_l + var_r)
                w.loc[left] = w.loc[left] * alpha
                w.loc[right] = w.loc[right] * (1.0 - alpha)
                new_clusters.extend([left, right])
            clusters = new_clusters
        return w

    def calculate_hrp_weights(self, selected_tickers: List[str]) -> pd.Series:
        """HRP allocator: correlation-distance → single-linkage dendrogram →
        quasi-diagonal leaf order → recursive inverse-cluster-variance bisection.
        Falls back to inverse-vol on singular/empty inputs."""
        if len(selected_tickers) == 1:
            return pd.Series([1.0], index=selected_tickers)

        try:
            window = min(252, len(self.prices))
            returns = self.prices[selected_tickers].tail(window).pct_change().dropna()
            if len(returns) < 2 or returns.shape[1] < 2:
                raise ValueError("insufficient return history for HRP")

            shrinkage_info = ""
            if self.cfg.use_ledoit_wolf:
                try:
                    lw = LedoitWolf().fit(returns.values)
                    cov_values = lw.covariance_
                    cov = pd.DataFrame(cov_values, index=returns.columns, columns=returns.columns)
                    # Derive correlation from the SHRUNK covariance so cov/corr stay consistent.
                    d = np.sqrt(np.diag(cov_values))
                    corr_values = cov_values / np.outer(d, d)
                    corr = pd.DataFrame(corr_values, index=returns.columns, columns=returns.columns).clip(-1.0, 1.0)
                    shrinkage_info = f" (LW shrinkage λ={lw.shrinkage_:.3f})"
                except Exception as lw_err:
                    print(f"⚠️ LedoitWolf failed ({lw_err}); using raw sample covariance.")
                    cov = returns.cov()
                    corr = returns.corr().clip(-1.0, 1.0)
            else:
                cov = returns.cov()
                corr = returns.corr().clip(-1.0, 1.0)
            dist = np.sqrt(0.5 * (1.0 - corr))
            link = linkage(squareform(dist.values, checks=False), method="single")
            order_idx = list(leaves_list(link))
            ordered = [selected_tickers[i] for i in order_idx]

            w = self._recursive_bisection(cov, ordered).reindex(selected_tickers)
            w = w / w.sum()

            inv = self._calculate_inverse_vol_weights(selected_tickers)
            print(f"🌳 HRP weights computed across {len(ordered)} cluster leaves{shrinkage_info}")
            print(f"   HRP : {w.round(4).to_dict()}")
            print(f"   IVol: {inv.round(4).to_dict()}")
            return w
        except Exception as e:
            print(f"⚠️ HRP failed ({e}); falling back to inverse-volatility weighting.")
            return self._calculate_inverse_vol_weights(selected_tickers)

    def calculate_kelly_weights(self, selected_tickers: List[str]) -> pd.Series:
        """Fractional Multi-Asset Kelly allocator (long-only, capped, normalized).
        F* = Σ⁻¹·μ over the last kelly_lookback days, scaled by kelly_fraction,
        clipped to [0, kelly_max_weight], then renormalized to sum to 1.0.
        Uses Ledoit-Wolf shrinkage for Σ and pinv for stability against singular Σ."""
        if len(selected_tickers) == 1:
            return pd.Series([1.0], index=selected_tickers)

        window = min(self.cfg.kelly_lookback, len(self.prices))
        returns = self.prices[selected_tickers].tail(window).pct_change().dropna()

        if len(returns) < 2 or returns.shape[1] < 2:
            print("⚠️ Kelly: insufficient return history; falling back to equal-weight.")
            return pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

        mu = returns.mean().values  # daily mean excess return (rf = 0%)

        shrinkage_info = ""
        try:
            lw = LedoitWolf().fit(returns.values)
            cov_values = lw.covariance_
            shrinkage_info = f" (LW shrinkage λ={lw.shrinkage_:.3f})"
        except Exception as lw_err:
            print(f"⚠️ Kelly LedoitWolf failed ({lw_err}); using raw sample covariance.")
            cov_values = returns.cov().values

        # F* = Σ⁻¹·μ — pinv (not inv) keeps us upright on singular covariance.
        cov_inv = np.linalg.pinv(cov_values)
        f_star = cov_inv @ mu

        # Fractional Kelly, long-only, per-name cap.
        f_frac = self.cfg.kelly_fraction * f_star
        f_frac = np.clip(f_frac, 0.0, self.cfg.kelly_max_weight)

        total = float(f_frac.sum())
        if total <= 0:
            print("⚠️ Kelly: all weights non-positive after clipping (negative μ basket); "
                  "falling back to equal-weight.")
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
        else:
            weights = pd.Series(f_frac / total, index=selected_tickers)

        print(f"📐 Kelly weights computed across {len(selected_tickers)} names "
              f"(fraction={self.cfg.kelly_fraction:.2f}, cap={self.cfg.kelly_max_weight:.2f}){shrinkage_info}")
        print(f"   Kelly: {weights.round(4).to_dict()}")
        return weights

    def calculate_weights(self, selected_tickers: List[str]) -> pd.Series:
        """Dispatcher: routes to Kelly or HRP per cfg.allocator. On the Kelly path,
        also computes HRP weights as a side-by-side diagnostic (logged, not used)."""
        if self.cfg.allocator == "kelly":
            print(f"💼 Allocator: KELLY (fraction={self.cfg.kelly_fraction:.2f})")
            kelly_w = self.calculate_kelly_weights(selected_tickers)
            print("   [Diagnostic] HRP weights for the same basket (NOT used):")
            _ = self.calculate_hrp_weights(selected_tickers)
            return kelly_w
        print("💼 Allocator: HRP")
        return self.calculate_hrp_weights(selected_tickers)

class DrawdownController:
    """Grossman-Zhou (1993) high-water-mark drawdown floor. Linearly scales risk
    from 1.0 at the peak M_t down to 0.0 when equity reaches α·M_t — a realized-
    drawdown guarantee that complements the HMM's forward-looking regime view."""

    def __init__(self, alpha: float, k: float, state_path: str):
        self.alpha = alpha
        self.k = k
        self.state_path = state_path

    def _load_state(self) -> Dict:
        try:
            with open(self.state_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ GZ state unreadable ({e}); seeding from current equity.")
            return {}

    def _save_state(self, hwm: float) -> None:
        try:
            with open(self.state_path, "w") as f:
                json.dump({"hwm": hwm, "last_updated": datetime.now(timezone.utc).isoformat()}, f)
        except OSError as e:
            print(f"⚠️ Could not persist GZ state to {self.state_path}: {e}")

    def update_and_get_exposure(self, current_equity: float) -> float:
        state = self._load_state()
        prev_hwm = float(state.get("hwm", current_equity))
        hwm = max(prev_hwm, current_equity)
        self._save_state(hwm)

        denom = hwm * (1.0 - self.alpha)
        if denom <= 0:
            return 1.0
        raw = (current_equity - self.alpha * hwm) / denom
        return float(np.clip(self.k * raw, 0.0, 1.0))


class ExecutionEngine:
    """
    Cash-only state machine:
      0. Cancel all open orders
      1. Build sell + buy baskets (buys stay in our pocket)
      2. Submit all sells (limit)
      3. Poll every 5s up to 60s; force market-sell stragglers
      4. Scale buy qtys to non_marginable_buying_power minus a cash buffer
      5. Submit limit buys
    """
    SELL_POLL_INTERVAL = 5   # seconds between status pings
    SELL_POLL_TIMEOUT  = 60  # total wait before forcing market sells
    CASH_BUFFER        = 0.01  # leave 1% of cash unspent

    # Terminal order states — anything else is still "in flight"
    _DONE = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
             OrderStatus.REJECTED, OrderStatus.REPLACED, OrderStatus.DONE_FOR_DAY}

    def __init__(self, config: StrategyConfig):
        self.cfg = config
        self.client = TradingClient(config.alpaca_key, config.alpaca_secret, paper=config.is_paper)
        self.gz: "DrawdownController | None" = None
        self._session_open_equity: float | None = None

    def attach_gz_controller(self, gz: "DrawdownController") -> None:
        """Attach the Grossman-Zhou controller post-construction so the breaker
        can consult the persisted HWM without reordering the run_strategy flow."""
        self.gz = gz

    def get_account_state(self) -> tuple[float, Dict[str, float], Dict[str, float]]:
        acct = self.client.get_account()
        equity = float(acct.equity)
        positions = self.client.get_all_positions()
        weights = {p.symbol: float(p.market_value) / equity for p in positions}
        qtys = {p.symbol: float(p.qty) for p in positions}
        return equity, weights, qtys

    # ---- Phase 0: cancel stale work ------------------------------------------------
    def _cancel_all_open_orders(self, dry_run: bool):
        if dry_run:
            print("🧹 [DRY] Would cancel all open orders.")
            return
        try:
            responses = self.client.cancel_orders()
            print(f"🧹 Cancelled {len(responses)} stale open order(s).")
        except Exception as e:
            print(f"⚠️ cancel_orders failed: {e}")

    # ---- Slicing helpers ----------------------------------------------------------
    def _compute_slice_schedule(self, total_qty: float, num_children: int,
                                convexity: float) -> List[float]:
        """Front-loaded convex schedule: weight_i ∝ (N - i)^convexity, normalized.
        Walks the remainder so the children sum to exactly total_qty after rounding
        (last child absorbs any drift). convexity=1.0 collapses to linear TWAP."""
        weights = np.array([(num_children - i) ** convexity for i in range(num_children)],
                           dtype=float)
        weights = weights / weights.sum()
        schedule, remaining = [], float(total_qty)
        for i in range(num_children - 1):
            q = min(round(float(total_qty) * float(weights[i]), 4), remaining)
            schedule.append(q)
            remaining = round(remaining - q, 4)
        schedule.append(round(remaining, 4))
        return schedule

    def _submit_one(self, sym: str, qty: float, limit_price, side, dry_run: bool,
                    side_label: str) -> List[str]:
        """Submit a single order. Returns [order_id] on success, [] otherwise."""
        if dry_run:
            tag = f"LMT ${limit_price:,.2f}" if limit_price is not None else "MKT (no price ref)"
            print(f"[DRY] {side_label:<5} | {sym:<5} | Qty: {qty:,.4f} | {tag}")
            return []
        try:
            if limit_price is None:
                req = MarketOrderRequest(symbol=sym, qty=qty, side=side,
                                         time_in_force=TimeInForce.DAY)
            else:
                req = LimitOrderRequest(symbol=sym, qty=qty, side=side,
                                        time_in_force=TimeInForce.DAY, limit_price=limit_price)
            order = self.client.submit_order(req)
            if side == OrderSide.BUY:
                print(f"[EXEC] BUY submitted: {sym} qty={qty} @ ${limit_price:,.2f}")
            else:
                print(f"[EXEC] SELL submitted: {sym} qty={qty}")
            return [str(order.id)]
        except Exception as e:
            print(f"[ERROR] {side_label} failed for {sym}: {e}")
            return []

    def _execute_potentially_sliced(self, sym: str, total_qty: float, limit_price,
                                    side, dry_run: bool, equity: float) -> List[str]:
        """Single entry point for one parent order. Slices it into a convex,
        front-loaded child schedule if it crosses the size threshold; otherwise
        submits as a single order. Returns the submitted order IDs."""
        side_label = "SELL" if side == OrderSide.SELL else "BUY"
        notional = total_qty * limit_price if limit_price is not None else 0.0
        should_slice = (
            self.cfg.slicing_enabled
            and limit_price is not None
            and total_qty > 0
            and self.cfg.slicing_num_children > 1
            and notional > self.cfg.slicing_threshold_pct * equity
        )

        if not should_slice:
            return self._submit_one(sym, total_qty, limit_price, side, dry_run, side_label)

        schedule = self._compute_slice_schedule(
            total_qty, self.cfg.slicing_num_children, self.cfg.slicing_convexity)
        display = [int(q) if q == int(q) else round(q, 4) for q in schedule]
        print(f"🔪 SLICING {side_label} {sym} {total_qty} → {len(schedule)} children: {display}")

        order_ids: List[str] = []
        sleep_seconds = (self.cfg.slicing_window_minutes * 60.0) / max(1, len(schedule) - 1)

        for i, child_qty in enumerate(schedule):
            if child_qty <= 0:
                continue
            # NOTE: each child reuses the parent limit price; live-quote repricing
            # per child is a future enhancement.
            ids = self._submit_one(sym, child_qty, limit_price, side, dry_run, side_label)
            order_ids.extend(ids)
            print(f"  └ Child {i + 1}/{len(schedule)} submitted: "
                  f"{sym} qty={child_qty} @ ${limit_price:,.2f}")
            if i < len(schedule) - 1:
                if dry_run:
                    print(f"  └ [DRY] Would sleep {sleep_seconds:.0f}s before next child")
                else:
                    # NOTE: blocking sleep is acceptable for the daily rebalance cadence.
                    # Move this to an async scheduler when intraday execution is added.
                    time.sleep(sleep_seconds)

        return order_ids

    # ---- Phase 1: build the plan ---------------------------------------------------
    def _build_order_plan(self, target_weights: pd.Series, current_weights: Dict[str, float],
                          current_qtys: Dict[str, float], equity: float,
                          latest_prices: pd.Series):
        """Return (sells, buys) where each entry is (symbol, qty, limit_price)."""
        sells, buys = [], []
        pad = self.cfg.limit_order_padding

        # Out-of-universe liquidations (drop-from-top-N): close full position
        for sym, qty in current_qtys.items():
            if sym in target_weights.index or qty <= 0:
                continue
            if sym not in latest_prices.index or pd.isna(latest_prices.loc[sym]):
                # No price reference — flag for market sell
                sells.append((sym, qty, None))
                continue
            price = float(latest_prices.loc[sym])
            sells.append((sym, qty, round(price * (1 - pad), 2)))

        # Rebalance: trims (sells) + new positions / top-ups (buys)
        for sym in target_weights.index:
            tgt_w = float(target_weights.loc[sym])
            cur_w = current_weights.get(sym, 0.0)
            drift = abs(tgt_w - cur_w)
            # Skip tiny drifts UNLESS target is 0 (then we're fully exiting)
            if drift < self.cfg.min_weight_drift and tgt_w > 0:
                continue
            if sym not in latest_prices.index or pd.isna(latest_prices.loc[sym]):
                continue
            price = float(latest_prices.loc[sym])
            delta_notional = (tgt_w - cur_w) * equity
            qty = round(abs(delta_notional) / price, 4)
            if qty <= 0:
                continue
            if delta_notional < 0:
                sells.append((sym, qty, round(price * (1 - pad), 2)))
            else:
                buys.append((sym, qty, round(price * (1 + pad), 2)))

        return sells, buys

    # ---- Phase 2: submit sells -----------------------------------------------------
    def _submit_sells(self, sells, dry_run: bool, equity: float) -> List[str]:
        order_ids = []
        for sym, qty, limit_price in sells:
            ids = self._execute_potentially_sliced(
                sym, qty, limit_price, OrderSide.SELL, dry_run, equity)
            order_ids.extend(ids)
        return order_ids

    # ---- Phase 3: wait for sells, force market on stragglers -----------------------
    def _wait_for_sells(self, order_ids: List[str], dry_run: bool):
        if dry_run or not order_ids:
            return

        pending = list(order_ids)
        elapsed = 0
        print(f"⏱️  Waiting up to {self.SELL_POLL_TIMEOUT}s for {len(pending)} sell(s) to fill...")

        while pending and elapsed < self.SELL_POLL_TIMEOUT:
            time.sleep(self.SELL_POLL_INTERVAL)
            elapsed += self.SELL_POLL_INTERVAL
            still_open = []
            for oid in pending:
                try:
                    order = self.client.get_order_by_id(oid)
                    if order.status in self._DONE:
                        continue
                    still_open.append(oid)
                except Exception as e:
                    print(f"⚠️ status check failed for {oid}: {e}")
                    still_open.append(oid)
            if len(still_open) < len(pending):
                print(f"  ✅ {len(pending) - len(still_open)} filled — {len(still_open)} still open at {elapsed}s")
            pending = still_open

        if not pending:
            print("✅ All sells confirmed done.")
            return

        # Stragglers: cancel + market sell the unfilled remainder
        print(f"⚡ Timeout: forcing market-sell on {len(pending)} hanging order(s).")
        market_ids = []
        for oid in pending:
            try:
                order = self.client.get_order_by_id(oid)
                self.client.cancel_order_by_id(oid)
                filled = float(order.filled_qty or 0)
                remaining = round(float(order.qty) - filled, 4)
                if remaining <= 0:
                    continue
                req = MarketOrderRequest(symbol=order.symbol, qty=remaining,
                                         side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                new_order = self.client.submit_order(req)
                market_ids.append(str(new_order.id))
                print(f"  ⚡ {order.symbol}: market-sell {remaining} (cancelled limit)")
            except Exception as e:
                print(f"⚠️ force-market-sell failed for {oid}: {e}")

        # Quick second wait for market orders to confirm
        if market_ids:
            for _ in range(6):  # up to ~15s
                time.sleep(2.5)
                still = []
                for oid in market_ids:
                    try:
                        o = self.client.get_order_by_id(oid)
                        if o.status not in self._DONE:
                            still.append(oid)
                    except Exception:
                        still.append(oid)
                market_ids = still
                if not market_ids:
                    break
            if market_ids:
                print(f"⚠️ {len(market_ids)} market sell(s) still unconfirmed — proceeding anyway.")

    # ---- Phase 4 & 5: scale and submit buys ---------------------------------------
    def _submit_buys_with_cash_budget(self, buys, dry_run: bool, equity: float):
        if not buys:
            print("\n(No buys queued.)")
            return

        basket_cost = sum(qty * price for _, qty, price in buys)

        if dry_run:
            print(f"\n[DRY] Buy basket notional: ${basket_cost:,.2f} (cash budget check skipped)")
            for sym, qty, price in buys:
                self._execute_potentially_sliced(
                    sym, qty, price, OrderSide.BUY, dry_run, equity)
            return

        try:
            acct = self.client.get_account()
            cash_available = float(acct.non_marginable_buying_power)
        except Exception as e:
            print(f"⚠️ Couldn't read non_marginable_buying_power, aborting buys: {e}")
            return

        budget = cash_available * (1 - self.CASH_BUFFER)
        print(f"\n💰 Non-marginable cash: ${cash_available:,.2f}  "
              f"| Budget after {self.CASH_BUFFER:.0%} buffer: ${budget:,.2f}  "
              f"| Buy basket: ${basket_cost:,.2f}")

        scale = 1.0
        if basket_cost > budget and basket_cost > 0:
            scale = budget / basket_cost
            print(f"⚖️  Scaling all buys by {scale:.4f} to fit cash budget.")

        for sym, qty, limit_price in buys:
            scaled_qty = round(qty * scale, 4)
            if scaled_qty <= 0:
                continue
            self._execute_potentially_sliced(
                sym, scaled_qty, limit_price, OrderSide.BUY, dry_run, equity)

    # ---- Circuit breaker (asymmetric: blocks buys, never sells) -------------------
    def _preflight_drawdown_check(self, dry_run: bool) -> Tuple[bool, str]:
        """Returns (allow_buys, reason). Sells are always allowed — the breaker
        only blocks NEW exposure. Two independent triggers:
          1. Trailing HWM drawdown (from the persisted GZ state) exceeds breaker_dd_from_hwm.
          2. Today's session drawdown (vs. equity at first call) exceeds breaker_dd_intraday.
        """
        if not self.cfg.breaker_enabled:
            return True, "breaker disabled"
        if dry_run:
            return True, "dry run — breaker informational only"

        try:
            equity = float(self.client.get_account().equity)
        except Exception as e:
            print(f"⚠️ Breaker: could not read equity ({e}); failing OPEN (allowing buys).")
            return True, f"equity read failed: {e}"

        # Trigger 1: trailing HWM drawdown
        if self.gz is not None:
            state = self.gz._load_state()
            hwm = float(state.get("hwm", equity))
            if hwm > 0:
                trailing_dd = max(0.0, (hwm - equity) / hwm)
                if trailing_dd >= self.cfg.breaker_dd_from_hwm:
                    return False, (f"trailing DD {trailing_dd:.2%} ≥ "
                                   f"limit {self.cfg.breaker_dd_from_hwm:.2%} "
                                   f"(HWM ${hwm:,.2f}, equity ${equity:,.2f})")

        # Trigger 2: intraday DD vs. first-seen-this-session equity
        if self._session_open_equity is None:
            self._session_open_equity = equity
        intraday_dd = max(0.0, (self._session_open_equity - equity) / self._session_open_equity)
        if intraday_dd >= self.cfg.breaker_dd_intraday:
            return False, (f"session DD {intraday_dd:.2%} ≥ "
                           f"limit {self.cfg.breaker_dd_intraday:.2%} "
                           f"(session open ${self._session_open_equity:,.2f}, equity ${equity:,.2f})")

        return True, f"OK (trailing DD acceptable, session DD {intraday_dd:.2%})"

    # ---- Orchestrator --------------------------------------------------------------
    def send_rebalance_orders(self, target_weights: pd.Series, latest_prices: pd.Series, dry_run: bool):
        # Circuit breaker preflight — runs BEFORE we cancel anything so the
        # decision is visible alongside the order plan log.
        allow_buys, reason = self._preflight_drawdown_check(dry_run)
        print(f"🚨 Circuit breaker: {'ALLOW' if allow_buys else 'BLOCK BUYS'} — {reason}")

        # Phase 0
        self._cancel_all_open_orders(dry_run)

        # Account snapshot
        try:
            equity, current_weights, current_qtys = self.get_account_state()
        except Exception as e:
            if dry_run:
                equity, current_weights, current_qtys = 100_000.0, {}, {}
            else:
                raise e
        print(f"\nAccount Equity: ${equity:,.2f}")

        # Phase 1
        sells, buys = self._build_order_plan(target_weights, current_weights,
                                             current_qtys, equity, latest_prices)
        print(f"📋 Plan: {len(sells)} sell(s), {len(buys)} buy(s).\n")

        # Asymmetric breaker enforcement: sells always proceed (so we can still
        # de-risk), but new buys are blocked until equity recovers.
        if not allow_buys and buys:
            print(f"🛑 Breaker tripped — dropping {len(buys)} planned buy(s); sells will still execute to de-risk.")
            buys = []

        # Phase 2 + 3
        # NOTE: when slicing is active on the sell side, _submit_sells itself
        # blocks for up to slicing_window_minutes (time.sleep between children),
        # so the "sells finish before buys begin" invariant is preserved without
        # extending SELL_POLL_TIMEOUT — the existing poll covers fills of the
        # last sell child after the window closes.
        sell_ids = self._submit_sells(sells, dry_run, equity)
        self._wait_for_sells(sell_ids, dry_run)

        # Phase 4 + 5
        self._submit_buys_with_cash_budget(buys, dry_run, equity)

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

def apply_sector_cap(
    ranked_tickers: List[str],
    sector_map: Dict[str, str],
    top_n: int,
    max_per_sector: int,
) -> List[str]:
    """Walk the ranking in score order; admit a ticker only if its sector bucket
    still has room. Tickers with no sector mapping are admitted freely (don't
    penalize names we couldn't classify). Returns up to top_n tickers respecting
    the per-sector cap."""
    selected: List[str] = []
    sector_counts: Dict[str, int] = {}
    for t in ranked_tickers:
        sector = sector_map.get(t)
        if sector is None:
            selected.append(t)
        else:
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue
            selected.append(t)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def run_strategy(dry_run: bool, enable_slicing: bool = False):
    cfg = StrategyConfig()
    if enable_slicing:
        cfg.slicing_enabled = True
    fetcher = DataFetcher()

    # --- PHASE 3: Dynamic Liquidity Screener ---
    dynamic_tickers, sector_map = fetcher.get_dynamic_universe(top_n=cfg.universe_size)
    cfg.universe = tuple(dynamic_tickers)

    # Ingest historical data (OHLC — opens drive the overnight/intraday features)
    ohlc = fetcher.fetch_daily_ohlc(list(cfg.universe) + ["SPY"], period="3y")
    prices = ohlc["Close"]
    opens = ohlc["Open"]

    # --- PHASE 2: HMM Risk Management ---
    hmm = MarketRegimeHMM(prices)
    regime = hmm.detect_regime()

    if regime == "BULL": exposure_hmm = cfg.exposure_bull
    elif regime == "NEUTRAL": exposure_hmm = cfg.exposure_neutral
    else: exposure_hmm = cfg.exposure_bear

    # --- Grossman-Zhou Drawdown Floor (layered with HMM) ---
    engine = ExecutionEngine(cfg)
    try:
        equity, _, _ = engine.get_account_state()
    except Exception as e:
        if dry_run:
            equity = 100_000.0
            print(f"[DRY] Seed equity ${equity:,.2f} for GZ controller ({e}).")
        else:
            raise

    gz = DrawdownController(cfg.drawdown_floor_alpha, cfg.drawdown_leverage_k, cfg.gz_state_path)
    exposure_gz = gz.update_and_get_exposure(equity)
    final_exposure = min(exposure_gz, exposure_hmm)
    print(f"[Risk] HMM exposure: {exposure_hmm:.2f} | GZ exposure: {exposure_gz:.2f} | Final: {final_exposure:.2f}")

    # Hand the GZ controller to the execution engine so its circuit breaker
    # can consult the persisted HWM at order-submission time.
    engine.attach_gz_controller(gz)

    # --- PHASE 1: AI Brain ---
    target_weights = pd.Series(0.0, index=cfg.universe)
    if final_exposure > 0:
        signals = MLSignalGenerator(cfg, prices, opens, sector_map=sector_map)
        scores = signals.calculate_ml_scores()
        ranked = scores.sort_values(ascending=False).index.tolist()
        if cfg.sector_cap_enabled and sector_map:
            top_tickers = apply_sector_cap(ranked, sector_map, cfg.top_n, cfg.max_names_per_sector)
            capped_sectors = {t: sector_map.get(t, "Unknown") for t in top_tickers}
            print(f"🏛️  Sector cap applied (max {cfg.max_names_per_sector}/sector): {capped_sectors}")
            if len(top_tickers) < cfg.top_n:
                print(f"⚠️ Sector cap left only {len(top_tickers)} name(s) "
                      f"(< top_n={cfg.top_n}); HRP will allocate across the reduced basket.")
        else:
            top_tickers = ranked[:cfg.top_n]

        optimizer = PortfolioOptimizer(cfg, prices)
        active_weights = optimizer.calculate_weights(top_tickers)

        for sym, w in active_weights.items():
            target_weights[sym] = w * final_exposure  # SCALED BY min(HMM, GZ)

    print("\nTarget Portfolio Allocations (Including Cash Reserves):")
    if final_exposure < 1.0:
        print(f"CASH   {1.0 - final_exposure:.4f}")
    print(target_weights[target_weights > 0].to_string())

    # Execution
    engine.send_rebalance_orders(target_weights, prices.iloc[-1], dry_run=dry_run)


def run_universe_audit():
    """Diagnostic: re-runs walk-forward IC over a broad pool (all current S&P 500),
    re-screening the top-N by training-window dollar volume per walk. Tests
    whether the inflated walk-forward IC under the today-screened panel is being
    driven by liquidity-rank look-ahead. Does NOT trade."""
    print("=" * 64)
    print("🔬 UNIVERSE AUDIT — testing for liquidity-rank look-ahead")
    print("=" * 64)

    cfg = StrategyConfig()
    fetcher = DataFetcher()

    # Pull the full current S&P 500 ticker list (still survivorship-biased to
    # today's index, but much broader than the today-top-50 panel).
    import io
    import requests
    header = {"User-Agent": "Mozilla/5.0"}
    html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=header).text
    table = pd.read_html(io.StringIO(html))
    df_sp500 = table[0]
    sp500_tickers = df_sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map_full = dict(zip(
        df_sp500['Symbol'].str.replace('.', '-', regex=False),
        df_sp500['GICS Sector']
    ))
    print(f"📋 Loaded {len(sp500_tickers)} S&P 500 names from Wikipedia")

    ohlc = fetcher.fetch_daily_ohlc(sp500_tickers + ["SPY"], period="3y")
    closes = ohlc["Close"]
    opens = ohlc["Open"]
    volumes = ohlc["Volume"]
    if volumes.empty:
        raise RuntimeError("Volume data missing from yfinance; cannot run audit.")

    # Restrict to tickers that have BOTH closes, opens, and volumes (intersection)
    common = closes.columns.intersection(opens.columns).intersection(volumes.columns)
    closes = closes[common]
    opens = opens[common]
    volumes = volumes[common]
    print(f"📊 Broad panel: {len(closes.columns)} tickers, {len(closes)} dates")

    # Per-date dollar volume (close × volume)
    dollar_vol = closes.mul(volumes, fill_value=np.nan)

    signals = MLSignalGenerator(cfg, closes, opens, sector_map=sector_map_full)
    feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "ma_slope",
                    "overnight_ret", "intraday_ret", "overnight_ret_5d",
                    "overnight_neg", "rv_w", "rv_m"]

    print("🚶 Running rescreened walk-forward...")
    wf = signals._compute_walkforward_ic_rescreened(
        closes, dollar_vol, feature_cols, top_n=cfg.universe_size
    )
    print("\n" + "=" * 64)
    print("AUDIT RESULT — walk-forward IC with per-walk training-window screening")
    print("=" * 64)
    print(f"   Walks:                       {wf['n_walks']}")
    print(f"   Test days:                   {wf['n_days']}")
    print(f"   Mean daily IC:               {wf['mean_ic']:.4f}")
    print(f"   Std daily IC:                {wf['std_ic']:.4f}")
    print(f"   Annualized IR:               {wf['ic_ir_ann']:.2f}")
    print(f"   Hit rate (IC > 0):           {wf['hit_rate']:.2%}")
    print(f"   Avg walk-to-walk universe Jaccard: "
          f"{wf.get('avg_walk_to_walk_universe_overlap', float('nan')):.2%}")
    print("=" * 64)
    print("Compare to today-screened walk-forward: mean IC ≈ 0.106, IR(ann) ≈ 7.65")
    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Quant Trader (Phase 3)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate targets without executing trades.")
    parser.add_argument("--enable-slicing", action="store_true",
                        help="Enable order slicing (front-loaded schedule for large orders).")
    parser.add_argument("--cmda", action="store_true",
                        help="Run Cluster-Based MDA feature-importance diagnostic and exit (no trading).")
    parser.add_argument("--audit-universe", action="store_true",
                        help="Run universe-selection look-ahead audit (broad S&P 500 + per-walk rescreening) and exit.")
    args = parser.parse_args()

    # Fail-safe to ensure keys are loaded if not passed through Streamlit
    if not args.cmda and not args.audit_universe and not os.getenv("APCA_API_KEY_ID"):
        print("🚨 Warning: No Alpaca API keys found in environment. Trades will fail unless --dry-run is used.")

    if args.audit_universe:
        run_universe_audit()
    elif args.cmda:
        print("=" * 64)
        print("🔬 cMDA DIAGNOSTIC MODE — no trades will be placed")
        print("=" * 64)
        cfg = StrategyConfig()
        fetcher = DataFetcher()
        dynamic_tickers, sector_map = fetcher.get_dynamic_universe(top_n=cfg.universe_size)
        cfg.universe = tuple(dynamic_tickers)
        ohlc = fetcher.fetch_daily_ohlc(list(cfg.universe) + ["SPY"], period="3y")
        prices = ohlc["Close"]
        opens = ohlc["Open"]
        signals = MLSignalGenerator(cfg, prices, opens, sector_map=sector_map)
        dataset = signals._engineer_features(prices)
        signals.run_cmda_diagnostic(dataset)
    else:
        run_strategy(dry_run=args.dry_run, enable_slicing=args.enable_slicing)