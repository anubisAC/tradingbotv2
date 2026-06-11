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
    MarketRegimeHMM,
    apply_sector_cap,
    PRODUCTION_FEATURES,
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
        use_hmm: bool = False,
        use_gz: bool = False,
        use_exit_discipline: bool = False,
        use_sector_cap: bool = False,
        use_sector_neutral_signal: bool = False,
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
            use_hmm: Scale each rebalance by the historical HMM regime exposure.
            use_gz: Scale each rebalance by a simulated Grossman-Zhou drawdown floor.
            use_exit_discipline: Apply time-stop and stop-loss exits during holding windows.
            use_sector_cap: Select the top basket with the live max-names-per-sector cap.
            use_sector_neutral_signal: Residualize predictions by sector before ranking.

        Returns:
            A dictionary containing performance metrics, attribution results,
            and daily portfolio data.
        """
        common_index = closes.index.intersection(opens.index).sort_values()
        common_columns = closes.columns.intersection(opens.columns)
        closes = closes.loc[common_index, common_columns].sort_index()
        opens = opens.loc[common_index, common_columns].sort_index()

        if "SPY" not in closes.columns:
            raise ValueError("Attribution backtest requires SPY in the close/open price data.")
        sector_map = {t: s for t, s in sector_map.items() if t in closes.columns}

        unique_dates = closes.index.sort_values().unique()
        rebalance_dates = unique_dates[::rebalance_freq]
        
        daily_returns = []
        exposure_rows = []
        exit_rows = []
        simulated_holdings = {}
        equity = 1.0
        gz_hwm = 1.0
        
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
            signal_config = StrategyConfig(**vars(config))
            signal_config.sector_neutralize_signal = bool(use_sector_neutral_signal)
            signals = MLSignalGenerator(signal_config, train_closes, train_opens, sector_map)
            
            # Train model
            feature_cols = PRODUCTION_FEATURES
            train_dataset = signals._engineer_features(train_closes)
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            model.fit(train_dataset[feature_cols], train_dataset["target_rank"])
            
            # Predict on rebalance date
            predict_closes = closes.loc[closes.index <= rebalance_date]
            predict_opens = opens.loc[opens.index <= rebalance_date]
            
            # Create a temporary signal generator to get features for prediction date.
            temp_signals = MLSignalGenerator(signal_config, predict_closes, predict_opens, sector_map)
            
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

                    required_values = {
                        "overnight_ret": overnight_ret_val,
                        "intraday_ret": intraday_ret_val,
                        "overnight_ret_5d": overnight_ret_5d_val,
                        "overnight_neg": overnight_neg_val,
                        "rv_w": rv_w_val,
                        "rv_m": rv_m_val,
                    }
                    if any(pd.isna(required_values[f]) for f in feature_cols if f in required_values):
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
                
            scores = pd.Series(
                model.predict(today_df[feature_cols]).astype(np.float64),
                index=today_df.index,
            )
            if signal_config.sector_neutralize_signal and sector_map:
                sectors_today = pd.Series({t: sector_map.get(t) for t in scores.index}).dropna()
                n_sectors = int(sectors_today.nunique())
                if len(sectors_today) > 0 and n_sectors > 1:
                    valid_tickers = sectors_today.index.tolist()
                    dummies = pd.get_dummies(sectors_today).astype(float)
                    y = scores.loc[valid_tickers].values.astype(np.float64)
                    try:
                        beta, *_ = np.linalg.lstsq(dummies.values, y, rcond=None)
                        scores.loc[valid_tickers] = y - dummies.values @ beta
                        print(f"Sector-neutralized signal: residualized across {n_sectors} sectors")
                    except np.linalg.LinAlgError as e:
                        print(f"WARNING: Sector residualization failed ({e}); using raw predictions.")

            # 2. Select top N and get weights
            ranked_tickers = scores.sort_values(ascending=False).index.tolist()
            if use_sector_cap and sector_map:
                top_tickers = apply_sector_cap(
                    ranked_tickers, sector_map, config.top_n, config.max_names_per_sector
                )
                print(f"Sector cap applied (max {config.max_names_per_sector}/sector): "
                      f"{ {t: sector_map.get(t, 'Unknown') for t in top_tickers} }")
            else:
                top_tickers = ranked_tickers[:config.top_n]

            if not top_tickers:
                continue

            selected_sectors = {sector_map.get(t) for t in top_tickers if sector_map.get(t)}
            cap_theoretical_max = (
                len(selected_sectors) * config.max_names_per_sector
                if use_sector_cap and sector_map else config.top_n
            )

            if allocation_method == "hrp":
                optimizer = PortfolioOptimizer(config, train_closes)
                weights = optimizer.calculate_hrp_weights(top_tickers)
            elif allocation_method == "equal":
                weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
            else:
                raise ValueError(f"Unknown allocation_method: {allocation_method}")

            hmm_exposure = 1.0
            regime = "DISABLED"
            hmm_covariance = "DISABLED"
            hmm_fit_note = ""
            if use_hmm:
                try:
                    hmm_model = MarketRegimeHMM(predict_closes, verbose=False)
                    regime = hmm_model.detect_regime()
                    hmm_covariance = getattr(hmm_model, "hmm_covariance_type", "unknown")
                    hmm_fit_note = getattr(hmm_model, "hmm_fit_note", "")
                    if regime == "BULL":
                        hmm_exposure = config.exposure_bull
                    elif regime == "NEUTRAL":
                        hmm_exposure = config.exposure_neutral
                    else:
                        hmm_exposure = config.exposure_bear
                except Exception as e:
                    print(f"WARNING: HMM regime failed on {rebalance_date.date()} ({e}); using 100% exposure.")
                    regime = "ERROR"
                    hmm_covariance = "ERROR"
                    hmm_fit_note = str(e)

            gz_exposure = 1.0
            if use_gz:
                gz_hwm = max(gz_hwm, equity)
                denom = gz_hwm * (1.0 - config.drawdown_floor_alpha)
                if denom > 0:
                    raw = (equity - config.drawdown_floor_alpha * gz_hwm) / denom
                    gz_exposure = float(np.clip(config.drawdown_leverage_k * raw, 0.0, 1.0))

            final_exposure = min(hmm_exposure, gz_exposure)
            weights = weights * final_exposure

            forced_sells = set()
            rebalance_prices = closes.loc[rebalance_date]
            if use_exit_discipline:
                for ticker, meta in list(simulated_holdings.items()):
                    if ticker not in rebalance_prices.index or pd.isna(rebalance_prices.loc[ticker]):
                        continue

                    latest_price = float(rebalance_prices.loc[ticker])
                    entry_price = float(meta["entry_price"])
                    days_held = (pd.Timestamp(rebalance_date) - pd.Timestamp(meta["entry_date"])).days
                    reason = None
                    if config.time_stop_days > 0 and days_held >= config.time_stop_days:
                        reason = "TIME_STOP"
                    elif entry_price > 0 and latest_price <= entry_price * (1.0 - config.stop_loss_pct):
                        reason = "STOP_LOSS"

                    if reason:
                        forced_sells.add(ticker)
                        exit_rows.append({
                            "date": rebalance_date,
                            "ticker": ticker,
                            "reason": reason,
                            "entry_date": meta["entry_date"],
                            "entry_price": entry_price,
                            "exit_price": latest_price,
                            "days_held": days_held,
                            "return_pct": (latest_price / entry_price - 1.0) * 100 if entry_price > 0 else np.nan,
                        })
                        del simulated_holdings[ticker]

            period_exit_count = len(forced_sells)
            if forced_sells:
                weights = weights.drop(index=[t for t in forced_sells if t in weights.index])

            positive_weights = weights[weights > 0].copy()
            for ticker in list(simulated_holdings.keys()):
                if ticker not in positive_weights.index:
                    del simulated_holdings[ticker]

            for ticker in positive_weights.index:
                if ticker not in simulated_holdings:
                    entry_price = rebalance_prices.get(ticker, np.nan)
                    if pd.notna(entry_price) and entry_price > 0:
                        simulated_holdings[ticker] = {
                            "entry_price": float(entry_price),
                            "entry_date": pd.Timestamp(rebalance_date),
                        }
            
            # 3. Simulate holding period
            return_mask = (closes.index > rebalance_date) & (closes.index <= hold_end_date)
            period_index = closes.loc[return_mask].index
            if len(period_index) == 0:
                continue

            if positive_weights.empty:
                port_returns = pd.Series(0.0, index=period_index)
            else:
                hold_mask = (closes.index >= rebalance_date) & (closes.index <= hold_end_date)
                hold_window = closes.loc[hold_mask, positive_weights.index]
                asset_returns = hold_window.pct_change().dropna()
                if asset_returns.empty:
                    continue

                adjusted = asset_returns.copy()
                if use_exit_discipline:
                    active = pd.Series(True, index=positive_weights.index)
                    for dt in adjusted.index:
                        adjusted.loc[dt, ~active] = 0.0
                        if pd.Timestamp(dt) >= pd.Timestamp(hold_end_date):
                            continue

                        latest_prices = hold_window.loc[dt]
                        deactivate_after_today = []
                        for ticker in active.index[active]:
                            meta = simulated_holdings.get(ticker)
                            if not meta:
                                continue

                            latest_price = latest_prices.get(ticker, np.nan)
                            if pd.isna(latest_price):
                                continue

                            entry_price = float(meta["entry_price"])
                            days_held = (pd.Timestamp(dt) - pd.Timestamp(meta["entry_date"])).days
                            reason = None
                            if config.time_stop_days > 0 and days_held >= config.time_stop_days:
                                reason = "TIME_STOP"
                            elif entry_price > 0 and latest_price <= entry_price * (1.0 - config.stop_loss_pct):
                                reason = "STOP_LOSS"

                            if reason:
                                deactivate_after_today.append((ticker, reason, float(latest_price), days_held))

                        for ticker, reason, latest_price, days_held in deactivate_after_today:
                            active.loc[ticker] = False
                            meta = simulated_holdings.pop(ticker, None)
                            entry_price = float(meta["entry_price"]) if meta else np.nan
                            entry_date = meta["entry_date"] if meta else pd.NaT
                            exit_rows.append({
                                "date": dt,
                                "ticker": ticker,
                                "reason": reason,
                                "entry_date": entry_date,
                                "entry_price": entry_price,
                                "exit_price": latest_price,
                                "days_held": days_held,
                                "return_pct": (latest_price / entry_price - 1.0) * 100
                                if pd.notna(entry_price) and entry_price > 0 else np.nan,
                            })
                            period_exit_count += 1
                port_returns = adjusted.dot(positive_weights.reindex(adjusted.columns).fillna(0.0))

            daily_returns.append(port_returns)
            equity *= float((1.0 + port_returns).prod())
            exposure_rows.append({
                "date": rebalance_date,
                "regime": regime,
                "hmm_covariance": hmm_covariance,
                "hmm_fit_note": hmm_fit_note,
                "hmm_exposure": hmm_exposure,
                "gz_exposure": gz_exposure,
                "final_exposure": final_exposure,
                "requested_top_n": config.top_n,
                "selected_names": len(top_tickers),
                "selected_sectors": len(selected_sectors),
                "sector_cap_theoretical_max": cap_theoretical_max,
                "gross_weight": float(positive_weights.sum()),
                "open_positions": len(simulated_holdings),
                "forced_exits": period_exit_count,
            })
            
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
                "exposures": pd.DataFrame(exposure_rows).set_index("date") if exposure_rows else pd.DataFrame(),
                "exit_events": pd.DataFrame(exit_rows),
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
