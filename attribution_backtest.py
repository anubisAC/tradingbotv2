"""
attribution_backtest.py
-----------------------
Long-only top-N backtest with Brinson-style return attribution to isolate
market beta, sector tilts, and residual stock-selection alpha.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from typing import Dict, List

from quant_trader import (
    StrategyConfig,
    DataFetcher,
    MLSignalGenerator,
    PortfolioOptimizer,
)


class AttributionBacktest:
    """
    Runs a walk-forward backtest and decomposes portfolio returns into
    market, sector, and stock-selection components.
    """

    def run(
        self,
        closes: pd.DataFrame,
        opens: pd.DataFrame,
        sector_map: Dict[str, str],
        config: StrategyConfig,
        rebalance_freq: int = 5,
        allocation_method: str = "hrp",
    ) -> Dict:
        """
        Executes the backtest and attribution.

        Args:
            closes: DataFrame of daily close prices for the universe.
            opens: DataFrame of daily open prices for the universe.
            sector_map: Dictionary mapping tickers to GICS sectors.
            config: StrategyConfig instance.
            rebalance_freq: Rebalance period in days.
            allocation_method: "hrp" or "equal".

        Returns:
            A dictionary containing performance metrics, attribution results,
            and daily portfolio data.
        """
        unique_dates = closes.index.sort_values().unique()
        rebalance_dates = unique_dates[::rebalance_freq]
        
        daily_returns = []
        
        # Walk-forward backtest
        print("Running walk-forward backtest...")
        for i in tqdm(range(len(rebalance_dates) - 1)):
            rebalance_date = rebalance_dates[i]
            hold_end_date = rebalance_dates[i+1]
            
            # --- AI Signal & Allocation ---
            train_mask = closes.index < rebalance_date
            train_closes = closes.loc[train_mask]
            train_opens = opens.loc[train_mask]
            
            if len(train_closes) < config.vol_lookback:
                continue
            
            # 1. Get AI scores
            signals = MLSignalGenerator(config, train_closes, train_opens, sector_map)
            
            # Train model
            feature_cols = [
                "ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "ma_slope",
                "overnight_ret", "intraday_ret", "overnight_ret_5d",
                "overnight_neg", "rv_w", "rv_m",
            ]
            train_dataset = signals._engineer_features(train_closes)
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            model.fit(train_dataset[feature_cols], train_dataset["target_rank"])
            
            # Predict on rebalance date
            predict_closes = closes.loc[closes.index <= rebalance_date]
            predict_opens = opens.loc[opens.index <= rebalance_date]
            
            # Create a temporary signal generator to get features for prediction date.
            temp_signals = MLSignalGenerator(config, predict_closes, predict_opens, sector_map)
            
            # Adapted logic from calculate_ml_scores to get features for a single historical date
            todays_data = []
            tickers = [c for c in predict_closes.columns if c != "SPY"]
            for ticker in tickers:
                p_raw = predict_closes[ticker].dropna()
                if len(p_raw) < 50: continue
                if ticker not in predict_opens.columns: continue
                o_raw = predict_opens[ticker].dropna()
                if len(o_raw) < 25: continue
                p = temp_signals._apply_savgol(p_raw)
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

                    if any(pd.isna(v) for v in [overnight_ret_val, intraday_ret_val,
                                                overnight_ret_5d_val, overnight_neg_val,
                                                rv_w_val, rv_m_val]):
                        continue
                except IndexError:
                    continue

                todays_features = {
                    "ticker": ticker, "ret_1d": p.pct_change(1).iloc[-1], "ret_5d": p.pct_change(5).iloc[-1],
                    "ret_20d": p.pct_change(20).iloc[-1], "vol_20d": p.pct_change(1).rolling(20).std().iloc[-1],
                    "rsi_14": temp_signals._compute_rsi(p, 14).iloc[-1],
                    "ma_slope": (p.rolling(20).mean().iloc[-1] / p.rolling(50).mean().iloc[-1]) - 1.0,
                    "overnight_ret": overnight_ret_val,
                    "intraday_ret": intraday_ret_val,
                    "overnight_ret_5d": overnight_ret_5d_val,
                    "overnight_neg": overnight_neg_val,
                    "rv_w": rv_w_val,
                    "rv_m": rv_m_val,
                }
                todays_data.append(todays_features)

            if not todays_data:
                continue

            today_df = pd.DataFrame(todays_data).set_index("ticker").dropna()
            if today_df.empty:
                continue
                
            scores = pd.Series(model.predict(today_df[feature_cols]), index=today_df.index)

            # 2. Select top N and get weights
            ranked_tickers = scores.sort_values(ascending=False).index.tolist()
            top_tickers = ranked_tickers[:config.top_n]

            if not top_tickers:
                continue

            if allocation_method == "hrp":
                optimizer = PortfolioOptimizer(config, train_closes)
                weights = optimizer.calculate_hrp_weights(top_tickers)
            elif allocation_method == "equal":
                weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
            else:
                raise ValueError(f"Unknown allocation_method: {allocation_method}")
            
            # 3. Simulate holding period
            hold_mask = (closes.index > rebalance_date) & (closes.index <= hold_end_date)
            asset_returns = closes.loc[hold_mask, top_tickers].pct_change().dropna()
            
            port_returns = asset_returns.dot(weights)
            daily_returns.append(port_returns)
            
        if not daily_returns:
            raise ValueError("Backtest produced no returns.")
            
        portfolio_returns = pd.concat(daily_returns).sort_index()
        
        # --- Performance Metrics ---
        print("\nCalculating performance and attribution...")
        equity_curve = (1 + portfolio_returns).cumprod()
        
        # Max Drawdown
        hwm = equity_curve.cummax()
        drawdown = (equity_curve - hwm) / hwm
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio
        annualized_return = portfolio_returns.mean() * 252
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

        # --- Attribution ---
        universe_returns = closes.pct_change().dropna()
        spy_returns = universe_returns["SPY"]
        
        # Create sector factor returns (equal-weighted baskets)
        sectors = sorted(list(set(s for s in sector_map.values() if s)))
        sector_returns = pd.DataFrame(index=universe_returns.index)
        for sector in sectors:
            sector_tickers = [t for t, s in sector_map.items() if s == sector and t in universe_returns.columns]
            if sector_tickers:
                sector_returns[sector] = universe_returns[sector_tickers].mean(axis=1)
        
        # Align data for regression
        data = pd.DataFrame({
            "portfolio": portfolio_returns,
            "market": spy_returns,
        }).join(sector_returns).dropna()
        
        y = data["portfolio"].values
        X_df = data.drop("portfolio", axis=1)
        X_df.insert(0, "intercept", 1.0) # Add constant for alpha
        
        # Regression: y = alpha + beta*market + sum(gamma*sector) + epsilon
        coeffs, residuals_sum_sq, _, _ = np.linalg.lstsq(X_df.values, y, rcond=None)
        
        # R-squared
        total_sum_sq = np.sum((y - y.mean())**2)
        r_squared = 1 - (residuals_sum_sq[0] / total_sum_sq) if total_sum_sq > 0 and len(residuals_sum_sq) > 0 else 0
        
        alpha_daily = coeffs[0]
        beta_spy = coeffs[1]
        sector_betas = pd.Series(coeffs[2:], index=X_df.columns[2:])
        
        return {
            "metrics": {
                "annualized_alpha_pct": alpha_daily * 252 * 100,
                "spy_beta": beta_spy,
                "r_squared": r_squared,
                "annualized_sharpe": sharpe_ratio,
                "max_drawdown_pct": max_drawdown * 100,
            },
            "sector_betas": sector_betas.to_dict(),
            "data": {
                "portfolio_returns": portfolio_returns,
                "equity_curve": equity_curve,
            }
        }

if __name__ == "__main__":
    cfg = StrategyConfig()
    fetcher = DataFetcher()

    print("--- Building Universe for Backtest ---")
    dynamic_tickers, sector_map = fetcher.get_dynamic_universe(top_n=cfg.universe_size)
    cfg.universe = tuple(dynamic_tickers)
    
    print("--- Fetching Historical Data ---")
    ohlc = fetcher.fetch_daily_ohlc(list(cfg.universe) + ["SPY"], period="3y")
    closes = ohlc["Close"]
    opens = ohlc["Open"]

    backtester = AttributionBacktest()
    
    print("\n--- Running HRP Backtest ---")
    results_hrp = backtester.run(closes, opens, sector_map, cfg, allocation_method="hrp")
    
    print("\n--- Running Equal-Weight Backtest ---")
    results_ew = backtester.run(closes, opens, sector_map, cfg, allocation_method="equal")

    # --- Comparison ---
    hrp_metrics = results_hrp["metrics"]
    ew_metrics = results_ew["metrics"]
    
    comp_df = pd.DataFrame({
        "HRP": {
            "Annualized Alpha (%)": hrp_metrics["annualized_alpha_pct"],
            "SPY Beta": hrp_metrics["spy_beta"],
            "R-Squared": hrp_metrics["r_squared"],
            "Annualized Sharpe": hrp_metrics["annualized_sharpe"],
            "Max Drawdown (%)": hrp_metrics["max_drawdown_pct"],
        },
        "Equal-Weight": {
            "Annualized Alpha (%)": ew_metrics["annualized_alpha_pct"],
            "SPY Beta": ew_metrics["spy_beta"],
            "R-Squared": ew_metrics["r_squared"],
            "Annualized Sharpe": ew_metrics["annualized_sharpe"],
            "Max Drawdown (%)": ew_metrics["max_drawdown_pct"],
        }
    })
    
    print("\n--- Allocator Comparison ---")
    print(comp_df.to_string(float_format="%.3f"))
    
    # Verdict
    hrp_sharpe = hrp_metrics["annualized_sharpe"]
    ew_sharpe = ew_metrics["annualized_sharpe"]
    
    print("\n--- Verdict ---")
    if ew_sharpe > hrp_sharpe:
        print("ALLOCATOR LEAK: HRP underperforms naive EW on the traded slice")
    else:
        print("HRP justified")
