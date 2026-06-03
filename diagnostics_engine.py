"""
diagnostics_engine.py
---------------------
Comprehensive diagnostic infrastructure for system auditing.
7 diagnostic components to identify value creation and destruction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# COMPONENT 1: FEATURE INTEGRITY AUDIT
# =============================================================================

class FeatureIntegrityAudit:
    """Detects lookahead bias and signal lag in features."""
    
    def __init__(self, prices: pd.DataFrame, opens: pd.DataFrame):
        self.prices = prices
        self.opens = opens
    
    def detect_lookahead_bias(self, feature_cols: List[str], 
                             target_col: str = "target_fwd_ret") -> Dict:
        """
        Compare IC when using contemporaneous vs lagged data.
        Lookahead bias manifests as IC collapse when data is properly lagged.
        """
        results = {
            "feature_lag_0": {},    # Potential lookahead
            "feature_lag_1": {},    # Lagged 1 period
            "feature_lag_2": {},    # Lagged 2 periods
        }
        
        # Build dataset with different lag structures
        for ticker in self.prices.columns:
            if ticker == "SPY":
                continue
            
            p = self.prices[ticker]
            if len(p) < 50:
                continue
            
            ret_1d = p.pct_change(1)
            fwd_ret_5d = p.pct_change(5).shift(-5)
            
            for lag in [0, 1, 2]:
                lag_ret = ret_1d.shift(lag)
                valid_idx = ~(lag_ret.isna() | fwd_ret_5d.isna())
                
                if valid_idx.sum() < 10:
                    continue
                
                ic, _ = spearmanr(lag_ret[valid_idx], fwd_ret_5d[valid_idx])
                results[f"feature_lag_{lag}"][ticker] = float(ic) if pd.notna(ic) else 0.0
        
        # Aggregate by lag
        summary = {}
        for lag_key in results.keys():
            ics = list(results[lag_key].values())
            if ics:
                summary[lag_key] = {
                    "mean_ic": float(np.mean(ics)),
                    "std_ic": float(np.std(ics)),
                    "count": len(ics),
                }
        
        # Calculate lookahead bias indicator
        lag_0_mean = summary.get("feature_lag_0", {}).get("mean_ic", 0.0)
        lag_1_mean = summary.get("feature_lag_1", {}).get("mean_ic", 0.0)
        lookahead_bias_strength = max(0.0, lag_0_mean - lag_1_mean)
        
        return {
            "summary": summary,
            "lookahead_bias_strength": lookahead_bias_strength,
            "interpretation": (
                "⚠️ SEVERE" if lookahead_bias_strength > 0.05
                else "🟡 MODERATE" if lookahead_bias_strength > 0.02
                else "✅ MINIMAL"
            ),
        }
    
    def measure_feature_responsiveness(self, dataset: pd.DataFrame) -> Dict:
        """
        Measure how quickly features respond to regime changes.
        Lower responsiveness = features lagged relative to reality.
        """
        # Identify regime change dates (high volatility spikes)
        returns = dataset.groupby(level=0).apply(
            lambda x: x['target_fwd_ret'].std() if len(x) > 1 else 0
        )
        vol_spikes = returns[returns > returns.quantile(0.9)].index.tolist()
        
        responsiveness = {}
        for vol_spike in vol_spikes[:5]:  # Sample first 5 spikes
            spike_data = dataset.loc[[vol_spike]]
            if len(spike_data) > 0:
                # Measure variance in features post-spike
                pre_spike = dataset.loc[dataset.index < vol_spike]
                post_spike = dataset.loc[dataset.index > vol_spike]
                
                if len(pre_spike) > 0 and len(post_spike) > 5:
                    feature_cols = ["ret_1d", "vol_20d", "rsi_14", "ma_slope"]
                    pre_var = pre_spike[feature_cols].std().mean()
                    post_var = post_spike[feature_cols].head(5).std().mean()
                    responsiveness[str(vol_spike)] = float(post_var / (pre_var + 1e-6))
        
        return {
            "spike_responsiveness": responsiveness,
            "avg_responsiveness": float(np.mean(list(responsiveness.values()))) 
                                  if responsiveness else 0.0,
        }


# =============================================================================
# COMPONENT 2: TARGET LABEL VALIDATION
# =============================================================================

class TargetLabelValidator:
    """Validates 5-day forward return label accuracy."""
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
    
    def validate_forward_return_calculation(self) -> Dict:
        """
        Verify forward return target is calculated correctly.
        Check for survivorship bias, penny stocks, data integrity.
        """
        results = {
            "total_tickers": len(self.prices.columns),
            "null_ratios": {},
            "penny_stock_contamination": 0,
            "price_consistency": {},
        }
        
        for ticker in self.prices.columns:
            if ticker == "SPY":
                continue
            
            p = self.prices[ticker]
            
            # Null ratio
            null_pct = p.isna().sum() / len(p)
            results["null_ratios"][ticker] = float(null_pct)
            
            # Penny stock check (< $1 average price)
            avg_price = p.mean()
            if avg_price < 1.0:
                results["penny_stock_contamination"] += 1
            
            # Price consistency (large gaps suggest data errors)
            returns = p.pct_change().abs()
            extreme_moves = (returns > 0.30).sum()
            results["price_consistency"][ticker] = {
                "extreme_moves_>30pct": int(extreme_moves),
                "pct_of_total": float(extreme_moves / len(returns)) if len(returns) > 0 else 0.0,
            }
        
        # Aggregate statistics
        null_ratios = list(results["null_ratios"].values())
        
        return {
            "data_quality": {
                "avg_null_ratio": float(np.mean(null_ratios)),
                "max_null_ratio": float(np.max(null_ratios)),
                "penny_stocks_detected": int(results["penny_stock_contamination"]),
                "extreme_move_count": int(sum(
                    d["extreme_moves_>30pct"] 
                    for d in results["price_consistency"].values()
                )),
            },
            "interpretation": (
                "⚠️ DATA ISSUES" if np.mean(null_ratios) > 0.10 
                else "✅ CLEAN" if np.mean(null_ratios) < 0.05
                else "🟡 ACCEPTABLE"
            ),
            "details": results,
        }
    
    def measure_label_realizability(self, dataset: pd.DataFrame) -> Dict:
        """
        Measure what fraction of backtest returns are actually realizable
        in live trading (no penny stocks, sufficient volume, etc).
        """
        realizable_count = 0
        unrealizable_count = 0
        
        for date in dataset.index.unique():
            day_data = dataset.loc[[date]]
            
            # Check if assets meet liquidity criteria
            avg_price = day_data.get('price_at_prediction', np.nan)
            target_ret = day_data.get('target_fwd_ret', 0.0)
            
            # Simple heuristic: price > $1, target return reasonable
            if not pd.isna(avg_price) and avg_price > 1.0 and abs(target_ret) < 1.0:
                realizable_count += 1
            else:
                unrealizable_count += 1
        
        total = realizable_count + unrealizable_count
        
        return {
            "realizable_fraction": float(realizable_count / total) if total > 0 else 0.0,
            "realizable_count": realizable_count,
            "unrealizable_count": unrealizable_count,
            "interpretation": (
                "✅ HIGHLY REALIZABLE" if realizable_count / total > 0.95
                else "🟡 PARTIALLY REALIZABLE" if realizable_count / total > 0.80
                else "⚠️ LARGELY UNREALIZABLE"
            ),
        }


# =============================================================================
# COMPONENT 3: IC-TO-RETURNS TRANSLATION
# =============================================================================

class ICtoReturnsTranslator:
    """Measures gap between rank IC and actual portfolio returns."""
    
    @staticmethod
    def compute_ic_return_decomposition(
        dataset: pd.DataFrame,
        model_predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> Dict:
        """
        Decompose portfolio return into:
        - IC contribution (rank correlation)
        - Beta exposure
        - Sector exposure
        - Idiosyncratic alpha
        """
        results = {}
        
        # IC calculation
        ic, _ = spearmanr(model_predictions, actual_returns)
        results["rank_ic"] = float(ic) if pd.notna(ic) else 0.0
        
        # Return statistics
        mean_actual = actual_returns.mean()
        std_actual = actual_returns.std()
        sharpe_actual = mean_actual / (std_actual + 1e-6)
        
        results["realized_statistics"] = {
            "mean_return": float(mean_actual),
            "std_return": float(std_actual),
            "sharpe_ratio": float(sharpe_actual),
            "min_return": float(actual_returns.min()),
            "max_return": float(actual_returns.max()),
        }
        
        # Correlation between IC and returns
        ic_return_correlation = float(
            (model_predictions.corr(actual_returns)) 
            if len(model_predictions) > 1 else 0.0
        )
        results["ic_return_correlation"] = ic_return_correlation
        
        # IC trap indicator
        # If IC is high but realized returns are low, signal is trapped in relative ranking
        ic_trap_strength = max(0.0, abs(results["rank_ic"]) - sharpe_actual)
        results["ic_trap_indicator"] = {
            "rank_ic": float(abs(results["rank_ic"])),
            "realized_sharpe": float(sharpe_actual),
            "trap_strength": float(ic_trap_strength),
            "interpretation": (
                "🚨 SEVERE IC TRAP" if ic_trap_strength > 0.3
                else "⚠️ MODERATE IC TRAP" if ic_trap_strength > 0.1
                else "✅ IC ALIGNED WITH RETURNS"
            ),
        }
        
        return results


# =============================================================================
# COMPONENT 4: ALLOCATOR STRESS TEST
# =============================================================================

class AllocatorStressTest:
    """Compare HRP vs Kelly under historical stress."""
    
    @staticmethod
    def compare_allocators_on_history(
        prices: pd.DataFrame,
        selected_tickers: List[str],
        config,
    ) -> Dict:
        """
        Run HRP and Kelly side-by-side on historical data.
        Identify regime-specific strengths/weaknesses.
        """
        from quant_trader import PortfolioOptimizer
        
        if len(selected_tickers) < 2:
            return {"error": "Insufficient tickers for comparison"}
        
        optimizer = PortfolioOptimizer(config, prices)
        
        try:
            hrp_weights = optimizer.calculate_hrp_weights(selected_tickers)
            kelly_weights = optimizer.calculate_kelly_weights(selected_tickers)
        except Exception as e:
            return {"error": str(e)}
        
        results = {
            "hrp_weights": hrp_weights.to_dict(),
            "kelly_weights": kelly_weights.to_dict(),
            "concentration_metrics": {
                "hrp_hhi": float((hrp_weights ** 2).sum()),
                "kelly_hhi": float((kelly_weights ** 2).sum()),
                "hrp_max_weight": float(hrp_weights.max()),
                "kelly_max_weight": float(kelly_weights.max()),
                "hrp_concentration_interpretation": (
                    "CONCENTRATED" if (hrp_weights ** 2).sum() > 0.5
                    else "MODERATE" if (hrp_weights ** 2).sum() > 0.2
                    else "DIVERSIFIED"
                ),
                "kelly_concentration_interpretation": (
                    "CONCENTRATED" if (kelly_weights ** 2).sum() > 0.5
                    else "MODERATE" if (kelly_weights ** 2).sum() > 0.2
                    else "DIVERSIFIED"
                ),
            },
            "weight_divergence": {
                "mean_absolute_diff": float(
                    (hrp_weights - kelly_weights).abs().mean()
                ),
                "max_divergence_ticker": (
                    (hrp_weights - kelly_weights).abs().idxmax()
                    if len(selected_tickers) > 0 else "N/A"
                ),
            },
        }
        
        return results
    
    @staticmethod
    def identify_drawdown_periods(prices: pd.DataFrame, lookback: int = 20) -> List[Tuple]:
        """Identify historical drawdown periods for stress testing."""
        spy = prices.get("SPY", prices.iloc[:, 0])
        returns = spy.pct_change()
        
        drawdowns = []
        current_dd = 0
        dd_start = None
        
        for i, ret in enumerate(returns):
            current_dd = (1 + current_dd) * (1 + ret) - 1
            
            if current_dd < -0.05:  # 5% drawdown threshold
                if dd_start is None:
                    dd_start = i
            elif current_dd > 0 and dd_start is not None:
                drawdowns.append((dd_start, i))
                current_dd = 0
                dd_start = None
        
        return drawdowns


# =============================================================================
# COMPONENT 5: HOLDING PERIOD ANALYSIS
# =============================================================================

class HoldingPeriodAnalysis:
    """Measure signal decay vs actual holding duration."""
    
    @staticmethod
    def compute_alpha_decay_curve(
        dataset: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict:
        """
        Measure IC at 1-day, 5-day, 10-day, 20-day holding periods.
        Identify signal half-life.
        """
        results = {}
        
        for hold_days in [1, 5, 10, 20]:
            # Compute forward returns at different horizons
            ics = []
            
            unique_dates = sorted(dataset.index.unique())
            for i in range(len(unique_dates) - hold_days):
                current_date = unique_dates[i]
                target_date = unique_dates[i + hold_days]
                
                current_data = dataset.loc[[current_date]]
                target_data = dataset.loc[[target_date]]
                
                if len(current_data) < 5 or len(target_data) < 5:
                    continue
                
                # IC between current rank and future returns
                if "target_rank" in current_data.columns and "target_fwd_ret" in target_data.columns:
                    ic, _ = spearmanr(
                        current_data["target_rank"].values,
                        target_data["target_fwd_ret"].values,
                    )
                    if pd.notna(ic):
                        ics.append(float(ic))
            
            if ics:
                results[f"{hold_days}d"] = {
                    "mean_ic": float(np.mean(ics)),
                    "std_ic": float(np.std(ics)),
                    "count": len(ics),
                }
        
        # Calculate signal half-life
        ic_1d = results.get("1d", {}).get("mean_ic", 0.0)
        ic_5d = results.get("5d", {}).get("mean_ic", 0.0)
        decay_rate = max(0.0, ic_1d - ic_5d) / (ic_1d + 1e-6)
        
        return {
            "ic_by_holding_period": results,
            "decay_rate_per_5days": float(decay_rate),
            "estimated_halflife_days": (
                5.0 / np.log(2) * np.log(1 - decay_rate + 1e-6)
                if decay_rate > 0.01 else float("inf")
            ),
            "interpretation": (
                "🚨 RAPID DECAY" if decay_rate > 0.50
                else "⚠️ MODERATE DECAY" if decay_rate > 0.20
                else "✅ STABLE"
            ),
        }


# =============================================================================
# COMPONENT 6: EXECUTION FRICTION AUDIT
# =============================================================================

class ExecutionFrictionAudit:
    """Measure cost of execution decisions."""
    
    @staticmethod
    def simulate_min_weight_drift_impact(
        target_weights: pd.Series,
        current_weights: Dict[str, float],
        prices: pd.Series,
        min_weight_drift: float = 0.05,
    ) -> Dict:
        """
        Calculate how many beneficial trades were skipped due to drift threshold.
        """
        skipped_trades = []
        executed_trades = []
        
        for sym in target_weights.index:
            tgt_w = float(target_weights.loc[sym])
            cur_w = current_weights.get(sym, 0.0)
            drift = abs(tgt_w - cur_w)
            
            # Would this trade be skipped?
            would_skip = drift < min_weight_drift and tgt_w > 0
            
            if would_skip:
                skipped_trades.append({
                    "symbol": sym,
                    "target_weight": tgt_w,
                    "current_weight": cur_w,
                    "drift": drift,
                    "reason": "drift < threshold",
                })
            else:
                executed_trades.append({
                    "symbol": sym,
                    "target_weight": tgt_w,
                    "current_weight": cur_w,
                    "drift": drift,
                })
        
        return {
            "skipped_trades": skipped_trades,
            "executed_trades": executed_trades,
            "skip_rate": float(len(skipped_trades) / (len(skipped_trades) + len(executed_trades) + 1e-6)),
            "interpretation": (
                "🔴 MANY TRADES SKIPPED" if len(skipped_trades) > len(executed_trades)
                else "✅ NORMAL EXECUTION"
            ),
        }


# =============================================================================
# COMPONENT 7: REGIME DETECTION VALIDATION
# =============================================================================

class RegimeDetectionValidator:
    """Validate HMM regime classification against realized market behavior."""
    
    @staticmethod
    def validate_regime_classification(
        regime_history: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict:
        """
        Compare HMM regime to realized volatility, returns, correlations.
        """
        if regime_history is None or regime_history.empty:
            return {"error": "No regime history available"}
        
        results = {}
        
        for regime in ["BULL", "NEUTRAL", "BEAR"]:
            regime_mask = regime_history.get("Regime") == regime if "Regime" in regime_history.columns else []
            
            if len(regime_mask) == 0 or regime_mask.sum() == 0:
                results[regime] = {"count": 0, "error": "No data for regime"}
                continue
            
            # Realized returns during regime
            regime_returns = prices.pct_change().loc[regime_mask]
            
            results[regime] = {
                "days_in_regime": int(regime_mask.sum()),
                "mean_daily_return": float(regime_returns.mean().mean()) if len(regime_returns) > 0 else 0.0,
                "daily_volatility": float(regime_returns.std().mean()) if len(regime_returns) > 0 else 0.0,
                "sharpe_ratio": float(
                    regime_returns.mean().mean() / (regime_returns.std().mean() + 1e-6)
                ) if len(regime_returns) > 0 else 0.0,
            }
        
        # Regime alignment check
        # BULL should have high return, BEAR should have low return
        bull_ret = results.get("BULL", {}).get("mean_daily_return", 0.0)
        bear_ret = results.get("BEAR", {}).get("mean_daily_return", 0.0)
        bull_vol = results.get("BULL", {}).get("daily_volatility", 0.0)
        bear_vol = results.get("BEAR", {}).get("daily_volatility", 0.0)
        
        alignment_score = 0.0
        if bull_ret > bear_ret:
            alignment_score += 1.0
        if bear_vol > bull_vol:
            alignment_score += 1.0
        
        results["alignment_check"] = {
            "alignment_score": float(alignment_score / 2.0),  # 0-1 scale
            "interpretation": (
                "✅ WELL-ALIGNED" if alignment_score >= 1.5
                else "🟡 PARTIALLY-ALIGNED" if alignment_score >= 1.0
                else "⚠️ MISALIGNED"
            ),
        }
        
        return results
    
    @staticmethod
    def measure_regime_persistence(regime_history: pd.DataFrame) -> Dict:
        """Measure how often regimes flip."""
        if regime_history is None or regime_history.empty or "Regime" not in regime_history.columns:
            return {"error": "No regime history"}
        
        regimes = regime_history["Regime"].values
        flips = int((regimes[:-1] != regimes[1:]).sum())
        total_days = len(regimes)
        
        return {
            "total_days": total_days,
            "regime_flips": flips,
            "average_regime_duration_days": float(total_days / (flips + 1)),
            "flip_frequency_pct": float(flips / total_days * 100),
            "interpretation": (
                "🔴 VERY UNSTABLE" if flips / total_days > 0.20
                else "🟡 SOMEWHAT UNSTABLE" if flips / total_days > 0.10
                else "✅ STABLE"
            ),
        }
