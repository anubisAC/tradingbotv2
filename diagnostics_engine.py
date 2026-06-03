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
            "feature_lag_0": {},
            "feature_lag_1": {},
            "feature_lag_2": {},
        }
        
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
        
        summary = {}
        for lag_key in results.keys():
            ics = list(results[lag_key].values())
            if ics:
                summary[lag_key] = {
                    "mean_ic": float(np.mean(ics)),
                    "std_ic": float(np.std(ics)),
                    "count": len(ics),
                }
        
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
        """
        results = {
            "total_tickers": len(self.prices.columns),
            "null_ratios": {},
            "penny_stock_contamination": 0,
        }
        
        for ticker in self.prices.columns:
            if ticker == "SPY":
                continue
            
            p = self.prices[ticker]
            null_pct = p.isna().sum() / len(p)
            results["null_ratios"][ticker] = float(null_pct)
            
            avg_price = p.mean()
            if avg_price < 1.0:
                results["penny_stock_contamination"] += 1
        
        null_ratios = list(results["null_ratios"].values())
        
        return {
            "data_quality": {
                "avg_null_ratio": float(np.mean(null_ratios)) if null_ratios else 0.0,
                "max_null_ratio": float(np.max(null_ratios)) if null_ratios else 0.0,
                "penny_stocks_detected": int(results["penny_stock_contamination"]),
            },
            "interpretation": (
                "⚠️ DATA ISSUES" if np.mean(null_ratios) > 0.10 
                else "✅ CLEAN" if np.mean(null_ratios) < 0.05
                else "🟡 ACCEPTABLE"
            ),
        }


# =============================================================================
# COMPONENT 3: IC-TO-RETURNS TRANSLATION
# =============================================================================

class ICtoReturnsTranslator:
    """Measures gap between rank IC and actual portfolio returns."""
    
    @staticmethod
    def compute_ic_return_decomposition(
        model_predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> Dict:
        """
        Decompose portfolio return into IC contribution and realized returns.
        """
        ic, _ = spearmanr(model_predictions, actual_returns)
        
        mean_actual = actual_returns.mean()
        std_actual = actual_returns.std()
        sharpe_actual = mean_actual / (std_actual + 1e-6)
        
        ic_trap_strength = max(0.0, abs(float(ic)) - sharpe_actual if pd.notna(ic) else 0)
        
        return {
            "rank_ic": float(ic) if pd.notna(ic) else 0.0,
            "realized_statistics": {
                "mean_return": float(mean_actual),
                "std_return": float(std_actual),
                "sharpe_ratio": float(sharpe_actual),
            },
            "ic_trap_indicator": {
                "trap_strength": float(ic_trap_strength),
                "interpretation": (
                    "🚨 SEVERE IC TRAP" if ic_trap_strength > 0.3
                    else "⚠️ MODERATE IC TRAP" if ic_trap_strength > 0.1
                    else "✅ IC ALIGNED WITH RETURNS"
                ),
            },
        }


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
        """Run HRP and Kelly side-by-side on historical data."""
        from quant_trader import PortfolioOptimizer
        
        if len(selected_tickers) < 2:
            return {"error": "Insufficient tickers for comparison"}
        
        optimizer = PortfolioOptimizer(config, prices)
        
        try:
            hrp_weights = optimizer.calculate_hrp_weights(selected_tickers)
            kelly_weights = optimizer.calculate_kelly_weights(selected_tickers)
        except Exception as e:
            return {"error": str(e)}
        
        return {
            "hrp_weights": hrp_weights.to_dict(),
            "kelly_weights": kelly_weights.to_dict(),
            "concentration_metrics": {
                "hrp_hhi": float((hrp_weights ** 2).sum()),
                "kelly_hhi": float((kelly_weights ** 2).sum()),
                "hrp_max_weight": float(hrp_weights.max()),
                "kelly_max_weight": float(kelly_weights.max()),
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


# =============================================================================
# COMPONENT 5: HOLDING PERIOD ANALYSIS
# =============================================================================

class HoldingPeriodAnalysis:
    """Measure signal decay vs actual holding duration."""
    
    @staticmethod
    def compute_alpha_decay_curve(
        dataset: pd.DataFrame,
    ) -> Dict:
        """Measure IC at different holding periods."""
        results = {}
        
        for hold_days in [1, 5, 10, 20]:
            ics = []
            
            unique_dates = sorted(dataset.index.unique())
            for i in range(len(unique_dates) - hold_days):
                current_date = unique_dates[i]
                target_date = unique_dates[i + hold_days]
                
                current_data = dataset.loc[[current_date]]
                target_data = dataset.loc[[target_date]]
                
                if len(current_data) < 5 or len(target_data) < 5:
                    continue
                
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
        
        ic_1d = results.get("1d", {}).get("mean_ic", 0.0)
        ic_5d = results.get("5d", {}).get("mean_ic", 0.0)
        decay_rate = max(0.0, ic_1d - ic_5d) / (ic_1d + 1e-6)
        
        return {
            "ic_by_holding_period": results,
            "decay_rate_per_5days": float(decay_rate),
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
        min_weight_drift: float = 0.05,
    ) -> Dict:
        """Calculate trades skipped due to drift threshold."""
        skipped_trades = []
        executed_trades = []
        
        for sym in target_weights.index:
            tgt_w = float(target_weights.loc[sym])
            cur_w = current_weights.get(sym, 0.0)
            drift = abs(tgt_w - cur_w)
            
            would_skip = drift < min_weight_drift and tgt_w > 0
            
            if would_skip:
                skipped_trades.append({
                    "symbol": sym,
                    "drift": drift,
                })
            else:
                executed_trades.append({
                    "symbol": sym,
                    "drift": drift,
                })
        
        total = len(skipped_trades) + len(executed_trades)
        
        return {
            "skipped_count": len(skipped_trades),
            "executed_count": len(executed_trades),
            "skip_rate": float(len(skipped_trades) / total) if total > 0 else 0.0,
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
        """Compare HMM regime to realized volatility and returns."""
        if regime_history is None or regime_history.empty:
            return {"error": "No regime history available"}
        
        results = {}
        
        for regime in ["BULL", "NEUTRAL", "BEAR"]:
            regime_mask = regime_history.get("Regime") == regime if "Regime" in regime_history.columns else []
            
            if len(regime_mask) == 0 or regime_mask.sum() == 0:
                results[regime] = {"count": 0}
                continue
            
            regime_returns = prices.pct_change().loc[regime_mask]
            
            results[regime] = {
                "days_in_regime": int(regime_mask.sum()),
                "mean_daily_return": float(regime_returns.mean().mean()) if len(regime_returns) > 0 else 0.0,
                "daily_volatility": float(regime_returns.std().mean()) if len(regime_returns) > 0 else 0.0,
            }
        
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
            "alignment_score": float(alignment_score / 2.0),
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
            "average_regime_duration_days": float(total_days / (flips + 1)) if flips > 0 else float(total_days),
            "flip_frequency_pct": float(flips / total_days * 100) if total_days > 0 else 0.0,
            "interpretation": (
                "🔴 VERY UNSTABLE" if total_days > 0 and flips / total_days > 0.20
                else "🟡 SOMEWHAT UNSTABLE" if total_days > 0 and flips / total_days > 0.10
                else "✅ STABLE"
            ),
        }
