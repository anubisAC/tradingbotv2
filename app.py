"""
app.py
------
Streamlit Dashboard for the Phase 3 AI Quant Trader.
Features: Live Portfolio Tracking, Risk Management Panel, Open Orders,
HMM Regime Timeline, Allocator Comparison, Slicing Preview, and
Dry-Run / Live Execute controls.
"""

import io
import os
import time
import json
import contextlib
import warnings
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest, GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from quant_trader import (
    StrategyConfig,
    DataFetcher,
    MLSignalGenerator,
    PortfolioOptimizer,
    MarketRegimeHMM,
    ExecutionEngine,
    DrawdownController,
    apply_sector_cap,
    run_universe_audit,
    run_feature_ablation_audit,
    PRODUCTION_FEATURES,
)
from diagnostics_engine import (
    FeatureIntegrityAudit,
    TargetLabelValidator,
    ICtoReturnsTranslator,
    AllocatorStressTest,
    HoldingPeriodAnalysis,
    ExecutionFrictionAudit,
    RegimeDetectionValidator
)
from attribution_backtest import AttributionBacktest


load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Phase 3 AI Alpha", layout="wide")
st.title("Phase 3: AI Quant Terminal")

# -----------------------------------------------------------------------------
# CONSTANTS / HELPERS
# -----------------------------------------------------------------------------
REGIME_FILL = {
    "BULL":    "rgba(34, 197, 94, 0.15)",   # green
    "NEUTRAL": "rgba(234, 179, 8, 0.15)",   # amber
    "BEAR":    "rgba(239, 68, 68, 0.15)",   # red
}
REGIME_ICON = {"BULL": "BULL", "NEUTRAL": "NEUTRAL", "BEAR": "BEAR"}
ALLOC_COLOR = {"HRP": "#3b82f6", "Kelly": "#f59e0b"}

SECTOR_COLORS = {
    "Information Technology": "#3b82f6", # blue
    "Health Care": "#ef4444",            # red
    "Financials": "#22c55e",             # green
    "Consumer Discretionary": "#f59e0b", # amber
    "Communication Services": "#8b5cf6", # purple
    "Industrials": "#64748b",            # slate
    "Consumer Staples": "#ec4899",       # pink
    "Energy": "#f97316",                 # orange
    "Utilities": "#06b6d4",              # cyan
    "Real Estate": "#14b8a6",            # teal
    "Materials": "#eab308",              # yellow
    "Unknown": "#9ca3af"                 # gray
}

@st.cache_data(ttl=3600*24)
def get_asset_metadata():
    """Fetch S&P 500 metadata (Company Name & GICS Sector) to color-code holdings."""
    import requests
    import io
    try:
        header = {"User-Agent": "Mozilla/5.0"}
        html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=header).text
        df = pd.read_html(io.StringIO(html))[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        return df.set_index('Symbol')[['Security', 'GICS Sector']].rename(
            columns={'Security': 'Name', 'GICS Sector': 'Sector'}
        ).to_dict(orient='index')
    except Exception:
        return {}


def compute_slice_schedule(total_qty: float, num_children: int, convexity: float):
    """Mirror of ExecutionEngine._compute_slice_schedule — pure, no API needed."""
    weights = np.array(
        [(num_children - i) ** convexity for i in range(num_children)], dtype=float
    )
    weights = weights / weights.sum()
    schedule, remaining = [], float(total_qty)
    for i in range(num_children - 1):
        q = min(round(float(total_qty) * float(weights[i]), 4), remaining)
        schedule.append(q)
        remaining = round(remaining - q, 4)
    schedule.append(round(remaining, 4))
    return schedule


def preview_breaker(equity: float, hwm: float, dd_hwm_limit: float, enabled: bool):
    """Pure-function breaker preview for the dashboard — no engine side effects."""
    if not enabled:
        return "⚪ OFF", "Circuit breaker is disabled in config.", "off"
    trailing_dd = max(0.0, (hwm - equity) / hwm) if hwm > 0 else 0.0
    if trailing_dd >= dd_hwm_limit:
        return (
            "🔴 BLOCK",
            f"Trailing DD {trailing_dd:.2%} ≥ limit {dd_hwm_limit:.2%}. "
            f"New buys would be refused.",
            "block",
        )
    return (
        "🟢 ALLOW",
        f"Trailing DD {trailing_dd:.2%} within limit. "
        f"Intraday DD checked at order-submission time.",
        "allow",
    )


def read_gz_state(path: Path, fallback_equity: float) -> float:
    """Read Grossman-Zhou HWM from disk; fall back to current equity if unset."""
    try:
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            return float(state.get("hwm", fallback_equity))
    except Exception:
        pass
    return fallback_equity


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.info(
    "**Session-only controls.** Sliders and toggles below tune this dashboard "
    "session — they don't change how the live bot runs. To persist a value, "
    "edit the `StrategyConfig` defaults in `quant_trader.py`.\n\n"
    "Alpaca credentials are loaded from `.env`."
)

st.sidebar.header("Strategy Parameters")
universe_size = st.sidebar.slider(
    "S&P 500 Screener Size", min_value=10, max_value=100, value=50, step=10,
    help="How many of the biggest, most actively traded S&P 500 stocks to analyze. 50 is the sweet spot.",
)
top_n = st.sidebar.slider(
    "Top-N Stocks to Buy", min_value=1, max_value=30, value=20,
    help="Out of the screened universe, how many to actually buy (highest AI scores). Higher breadth helps realize weaker signals.",
)
vol_lookback = st.sidebar.slider(
    "Volatility Lookback (days)", min_value=10, max_value=60, value=30,
    help="History window used to gauge stock stability for volatility weighting.",
)
max_names_per_sector = st.sidebar.slider(
    "Max Names per Sector", min_value=1, max_value=10, value=2, step=1,
    help="Sector cap used when selecting the final basket. Higher values allow more names from the same GICS sector.",
)

st.sidebar.divider()
st.sidebar.header("Allocator")
allocator_label = st.sidebar.radio(
    "Weight Allocation Method", ["HRP", "Kelly"], horizontal=True,
    help="HRP = Hierarchical Risk Parity (López de Prado). "
         "Kelly = Fractional Multi-Asset Kelly. Toggle re-uses cached weights without re-training.",
)
kelly_fraction = st.sidebar.slider(
    "Kelly Fraction", min_value=0.05, max_value=1.00, value=0.25, step=0.05,
    help="Multiplier on the full Kelly weight. 0.25 = Quarter Kelly (more conservative).",
    disabled=(allocator_label != "Kelly"),
)
kelly_max_weight = st.sidebar.slider(
    "Kelly Max Weight / Name", min_value=0.05, max_value=1.00, value=0.40, step=0.05,
    help="Per-name cap to prevent absurd concentration on a single stock.",
    disabled=(allocator_label != "Kelly"),
)

with st.sidebar.expander("Risk Controls"):
    drawdown_floor_alpha = st.slider(
        "GZ Floor α", min_value=0.50, max_value=0.95, value=0.85, step=0.01,
        help="Grossman-Zhou drawdown floor. 0.85 = exposure linearly scales to 0 at 15% DD from HWM.",
    )
    breaker_dd_from_hwm = st.slider(
        "Breaker DD from HWM", min_value=0.02, max_value=0.20, value=0.07, step=0.01,
        help="If trailing drawdown from HWM exceeds this, the circuit breaker blocks new buys.",
    )
    breaker_dd_intraday = st.slider(
        "Breaker Intraday DD", min_value=0.01, max_value=0.10, value=0.03, step=0.005,
        help="If today's session drawdown exceeds this, the circuit breaker blocks new buys.",
    )
    breaker_enabled = st.checkbox("Breaker Enabled", value=True)
    min_weight_drift = st.slider(
        "Min Weight Drift", min_value=0.00, max_value=0.10, value=0.02, step=0.005,
        help="Rebalance trades whose target-vs-current weight gap is below this "
             "are skipped. Lower = more small trades executed (higher turnover).",
    )

with st.sidebar.expander("Execution Slicing"):
    slicing_enabled = st.checkbox("Slicing Enabled", value=False)
    slicing_convexity = st.slider(
        "Convexity", min_value=1.0, max_value=4.0, value=2.0, step=0.5,
        help="1.0 = linear TWAP. Higher values front-load the schedule (more shares earlier).",
    )
    slicing_num_children = st.slider(
        "Children per Slice", min_value=2, max_value=10, value=4, step=1,
    )
    slicing_threshold_pct = st.slider(
        "Slice Threshold (% of Equity)", min_value=0.01, max_value=0.20,
        value=0.05, step=0.01,
        help="Orders above this fraction of equity get sliced. Below it: single shot.",
    )
    slicing_window_minutes = st.slider(
        "Slicing Window (min)", min_value=5, max_value=120, value=5, step=5,
    )

# Resolve Keys
api_key = os.getenv("APCA_API_KEY_ID")
api_secret = os.getenv("APCA_API_SECRET_KEY")
is_paper = os.getenv("APCA_PAPER", "true").lower() == "true"


def make_config() -> StrategyConfig:
    """Build a StrategyConfig from current sidebar state."""
    cfg = StrategyConfig(
        top_n=top_n,
        vol_lookback=vol_lookback,
        universe_size=universe_size,
        max_names_per_sector=int(max_names_per_sector),
        allocator=allocator_label.lower(),
        kelly_fraction=float(kelly_fraction),
        kelly_max_weight=float(kelly_max_weight),
        drawdown_floor_alpha=float(drawdown_floor_alpha),
        breaker_dd_from_hwm=float(breaker_dd_from_hwm),
        breaker_dd_intraday=float(breaker_dd_intraday),
        breaker_enabled=bool(breaker_enabled),
        min_weight_drift=float(min_weight_drift),
        slicing_enabled=bool(slicing_enabled),
        slicing_convexity=float(slicing_convexity),
        slicing_num_children=int(slicing_num_children),
        slicing_threshold_pct=float(slicing_threshold_pct),
        slicing_window_minutes=int(slicing_window_minutes),
    )
    if api_key:
        cfg.alpaca_key = api_key
    if api_secret:
        cfg.alpaca_secret = api_secret
    cfg.is_paper = is_paper
    return cfg


# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab_portfolio, tab_research, tab_diagnostics, tab_system_audit = st.tabs([
    "Live Portfolio Performance",
    "AI Research & Manual Execution",
    "Diagnostics",
    "System Audit Engine",
])

# =============================================================================
# TAB 1: LIVE PORTFOLIO
# =============================================================================
with tab_portfolio:
    if not api_key or not api_secret:
        st.warning(
            "⚠️ Alpaca API keys not found. Set `APCA_API_KEY_ID` and "
            "`APCA_API_SECRET_KEY` in your `.env` file to view live portfolio data."
        )
    else:
        try:
            client = TradingClient(api_key, api_secret, paper=is_paper)
            account = client.get_account()

            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pnl = equity - last_equity
            daily_pct = (daily_pnl / last_equity) if last_equity > 0 else 0.0
            cash = float(account.cash)
            cash_weight = cash / equity if equity > 0 else 0.0

            st.subheader("Performance Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Total Equity", f"${equity:,.2f}", f"{daily_pnl:+,.2f} Today",
                help="Total real-world account value right now (cash + current value of all stocks).",
            )
            m2.metric(
                "Daily Return", f"{daily_pct:.2%}",
                help="How much your portfolio moved today.",
            )
            m3.metric(
                "Purchasing Power", f"${float(account.buying_power):,.2f}",
                help="Buying power Alpaca extends (may include margin). The bot does not use margin.",
            )
            m4.metric(
                "Cash Drag", f"{cash_weight:.2%}",
                help="% of equity sitting in cash. High cash drag protects you in bear markets.",
            )

            # ---- NEW: Risk Management Panel ----
            st.divider()
            st.subheader("Risk Management")
            with st.container(border=True):
                gz_path = Path(StrategyConfig().gz_state_path)
                hwm = read_gz_state(gz_path, equity)
                trailing_dd = max(0.0, (hwm - equity) / hwm) if hwm > 0 else 0.0

                # GZ exposure cap (peek-only — does NOT update the persisted HWM)
                denom = hwm * (1.0 - drawdown_floor_alpha)
                if denom <= 0:
                    gz_exposure = 1.0
                else:
                    gz_exposure = float(
                        np.clip((equity - drawdown_floor_alpha * hwm) / denom, 0.0, 1.0)
                    )

                breaker_label, breaker_reason, breaker_kind = preview_breaker(
                    equity, hwm, breaker_dd_from_hwm, breaker_enabled
                )

                g1, g2, g3, g4 = st.columns(4)
                g1.metric(
                    "High-Water Mark", f"${hwm:,.2f}",
                    help="Highest equity ever recorded by the bot. Drives the GZ drawdown floor.",
                )
                g2.metric(
                    "Drawdown from HWM",
                    f"{trailing_dd:.2%}",
                    delta=f"-{trailing_dd:.2%}" if trailing_dd > 0 else "0.00%",
                    delta_color="inverse",
                    help="How far below the HWM equity currently sits.",
                )
                g3.metric(
                    "GZ Exposure Cap", f"{gz_exposure:.0%}",
                    help="Drawdown-floor risk multiplier. Combined with HMM regime as min(both).",
                )
                g4.metric("Circuit Breaker", breaker_label, help=breaker_reason)

                if breaker_kind == "block":
                    st.error(f"BREAKER: {breaker_reason}")
                elif breaker_kind == "allow":
                    st.success(f"BREAKER: {breaker_reason}")
                else:
                    st.info(f"BREAKER: {breaker_reason}")

            # ---- Equity curve ----
            st.divider()
            st.subheader("1-Month Equity Curve")
            try:
                hist_req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
                history = client.get_portfolio_history(hist_req)
                if history.timestamp:
                    dates = [pd.to_datetime(ts, unit='s').date() for ts in history.timestamp]
                    eq_df = pd.DataFrame({"Date": dates, "Equity": history.equity})
                    fig_eq = px.line(eq_df, x="Date", y="Equity", title="Account Equity Over Time")
                    fig_eq.update_layout(yaxis_title="USD ($)", xaxis_title="")
                    st.plotly_chart(fig_eq, use_container_width=True)
                else:
                    st.info("Not enough historical data generated yet to display equity curve.")
            except Exception as e:
                st.warning(f"Could not load equity history: {e}")

            # ---- Current Positions ----
            st.subheader("Active AI Holdings")
            positions = client.get_all_positions()
            if positions:
                meta = get_asset_metadata()
                pos_data = []
                for p in positions:
                    info = meta.get(p.symbol, {"Name": "Unknown", "Sector": "Unknown"})
                    pos_data.append({
                        "Ticker": p.symbol,
                        "Name": info["Name"],
                        "Sector": info["Sector"],
                        "Qty": float(p.qty),
                        "Market Value": float(p.market_value),
                        "Unrealized PnL": float(p.unrealized_pl),
                        "Weight": float(p.market_value) / equity if equity > 0 else 0.0,
                    })
                pos_df = pd.DataFrame(pos_data).sort_values(by="Weight", ascending=False)

                # Build a dynamic aesthetic key for active sectors
                active_sectors = pos_df["Sector"].unique()
                key_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; font-size: 0.85rem; font-weight: 500;'>"
                for sec in active_sectors:
                    color = SECTOR_COLORS.get(sec, SECTOR_COLORS["Unknown"])
                    key_html += f"<span style='background-color: {color}15; color: {color}; border: 1px solid {color}40; padding: 4px 10px; border-radius: 6px;'>{sec}</span>"
                key_html += "</div>"
                st.markdown(key_html, unsafe_allow_html=True)

                # Styling logic for Pandas
                def style_sector(val):
                    color = SECTOR_COLORS.get(val, SECTOR_COLORS["Unknown"])
                    return f'color: {color}; font-weight: 600;'

                # Keep Name in the display dataframe; Streamlit doesn't support HTML cell tooltips
                display_df = pos_df[["Ticker", "Name", "Sector", "Qty", "Market Value", "Unrealized PnL", "Weight"]]

                styled_df = (display_df.style
                    .format({
                        "Qty": "{:,.4f}",
                        "Market Value": "${:,.2f}",
                        "Unrealized PnL": "${:,.2f}",
                        "Weight": "{:.2%}",
                    })
                    .map(style_sector, subset=["Sector"])
                )

                st.dataframe(styled_df, width="stretch", hide_index=True)
                port_markdown = display_df.to_markdown(index=False)
            else:
                st.info("Portfolio is currently 100% Cash. Waiting for AI deployment.")
                port_markdown = "*No active positions. 100% Cash.*"

            # ---- NEW: Open Orders + Cancel All ----
            st.divider()
            st.subheader("Open Orders")
            try:
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                open_orders = client.get_orders(filter=req)
            except Exception as e:
                open_orders = []
                st.warning(f"Could not fetch open orders: {e}")

            if open_orders:
                orders_data = []
                for o in open_orders:
                    orders_data.append({
                        "Symbol": o.symbol,
                        "Side": str(o.side.value if hasattr(o.side, 'value') else o.side).upper(),
                        "Qty": float(o.qty),
                        "Type": str(o.order_type.value if hasattr(o.order_type, 'value') else o.order_type).upper(),
                        "Limit": float(o.limit_price) if o.limit_price else None,
                        "Status": str(o.status.value if hasattr(o.status, 'value') else o.status).upper(),
                        "Submitted": pd.to_datetime(o.submitted_at).strftime("%Y-%m-%d %H:%M") if o.submitted_at else "—",
                    })
                orders_df = pd.DataFrame(orders_data)
                st.dataframe(
                    orders_df.style.format({"Qty": "{:,.4f}", "Limit": "${:,.2f}"}, na_rep="—"),
                    width="stretch", hide_index=True,
                )

                with st.container(border=True):
                    co1, co2 = st.columns([2, 1])
                    confirm_cancel = co1.checkbox(
                        "I want to cancel ALL open orders", key="confirm_cancel_all_orders",
                    )
                    if co2.button(
                        "Cancel All", disabled=not confirm_cancel,
                        type="secondary", use_container_width=True,
                    ):
                        try:
                            target_ids = {str(o.id) for o in open_orders}
                            responses = client.cancel_orders()
                            with st.spinner("Cancelling and waiting for confirmation…"):
                                deadline = time.monotonic() + 5.0
                                remaining = set(target_ids)
                                while remaining and time.monotonic() < deadline:
                                    time.sleep(0.5)
                                    try:
                                        still_open = client.get_orders(
                                            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
                                        )
                                        open_ids = {str(o.id) for o in still_open}
                                        remaining = target_ids & open_ids
                                    except Exception:
                                        break
                            if remaining:
                                st.warning(
                                    f"Submitted cancellation for {len(responses)} order(s). "
                                    f"{len(remaining)} still showing OPEN — refresh in a moment."
                                )
                            else:
                                st.success(f"Cancelled {len(responses)} order(s).")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cancel failed: {e}")
            else:
                st.info("No open orders.")

            # ---- Export Live Portfolio ----
            st.divider()
            st.subheader("Export Live Portfolio")
            st.caption("Hover the box and click 'Copy' to share status with an AI.")
            report = f"**LIVE PORTFOLIO STATUS**\n"
            report += f"- **Total Equity:** ${equity:,.2f} ({daily_pnl:+,.2f} Today | {daily_pct:.2%})\n"
            report += f"- **Purchasing Power:** ${float(account.buying_power):,.2f}\n"
            report += f"- **Cash Drag:** {cash_weight:.2%}\n"
            report += f"- **HWM:** ${hwm:,.2f}  |  **DD from HWM:** {trailing_dd:.2%}  |  **GZ Cap:** {gz_exposure:.0%}\n"
            report += f"- **Circuit Breaker:** {breaker_label} — {breaker_reason}\n\n"
            report += "**Current Holdings:**\n"
            report += port_markdown
            st.code(report, language="markdown")

        except Exception as e:
            st.error(f"Failed to connect to Alpaca: {e}")

# =============================================================================
# TAB 2: AI RESEARCH TERMINAL
# =============================================================================
with tab_research:
    run_pipeline = st.button("Run Manual AI Calculation", type="primary")

    if run_pipeline:
        cfg = make_config()
        if not cfg.alpaca_key or not cfg.alpaca_secret:
            st.error(
                "Alpaca credentials not found in `.env`. "
                "Set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` and restart the dashboard."
            )
            st.stop()

        with st.spinner(f"Scanning S&P 500 for the top {cfg.universe_size} most liquid stocks..."):
            fetcher = DataFetcher()
            dynamic_tickers, sector_map = fetcher.get_dynamic_universe(top_n=cfg.universe_size)
            cfg.universe = tuple(dynamic_tickers)

        with st.spinner("Fetching 3 years of market data..."):
            ohlc = fetcher.fetch_daily_ohlc(list(cfg.universe) + ["SPY"], period="3y")
            prices = ohlc["Close"]
            opens = ohlc["Open"]

        with st.spinner("Running Hidden Markov Model for Regime Detection..."):
            hmm_model = MarketRegimeHMM(prices)
            current_regime = hmm_model.detect_regime()

        if current_regime == "BULL":
            exposure = cfg.exposure_bull
        elif current_regime == "NEUTRAL":
            exposure = cfg.exposure_neutral
        else:
            exposure = cfg.exposure_bear

        # Defaults for BEAR-regime / no-trading path
        signals = None
        scores_df = pd.DataFrame()
        top_tickers: list = []
        hrp_weights = pd.Series(dtype=float)
        kelly_weights = pd.Series(dtype=float)

        if exposure > 0:
            with st.spinner("Engineering features and training XGBoost..."):
                signals = MLSignalGenerator(cfg, prices, opens, sector_map=sector_map)
                scores = signals.calculate_ml_scores()

            scores_df = scores.reset_index()
            scores_df.columns = ["Ticker", "Predicted Rank"]
            scores_df = scores_df.sort_values(by="Predicted Rank", ascending=False)

            ranked = scores_df["Ticker"].tolist()
            if cfg.sector_cap_enabled and sector_map:
                top_tickers = apply_sector_cap(
                    ranked, sector_map, cfg.top_n, cfg.max_names_per_sector
                )
            else:
                top_tickers = ranked[:cfg.top_n]

            with st.spinner("⚖️ Computing HRP and Kelly weights (cached for hot-swap)..."):
                optimizer = PortfolioOptimizer(cfg, prices)
                hrp_weights = optimizer.calculate_hrp_weights(top_tickers)
                kelly_weights = optimizer.calculate_kelly_weights(top_tickers)

        # Stash everything in session_state so other widgets can re-render
        # without triggering a re-run of the expensive pipeline.
        st.session_state.plan = {
            "config_at_run": cfg,
            "regime": current_regime,
            "exposure": exposure,
            "regime_history": getattr(hmm_model, "regime_history", None),
            "ic_metrics": getattr(signals, "last_ic_metrics", None) if signals else None,
            "scores_df": scores_df,
            "top_tickers": top_tickers,
            "sector_map": sector_map,
            "hrp_weights": hrp_weights,
            "kelly_weights": kelly_weights,
            "latest_prices": prices.iloc[-1] if not prices.empty else pd.Series(dtype=float),
            "spy_prices": prices["SPY"].copy() if "SPY" in prices.columns else pd.Series(dtype=float),
            "full_prices": prices.copy() if not prices.empty else pd.DataFrame(),
            "computed_at": pd.Timestamp.now(),
        }

    # ---------------------------------------------------------------- RENDER
    if "plan" not in st.session_state:
        st.info("👆 Click **Run Manual AI Calculation** above to generate today's execution plan.")
    else:
        plan = st.session_state.plan
        cfg_run = plan["config_at_run"]
        current_regime = plan["regime"]
        exposure = plan["exposure"]

        # Apply current sidebar params on top of plan (allocator hot-swap, etc.)
        cfg_live = make_config()
        cfg_live.universe = cfg_run.universe  # universe is fixed at Run time

        icon = REGIME_ICON.get(current_regime, "")
        st.divider()
        st.caption(
            f"Plan computed at **{plan['computed_at'].strftime('%Y-%m-%d %H:%M:%S')}** "
            f"| Universe: **{len(cfg_run.universe)}** stocks "
            f"| Allocator (live): **{cfg_live.allocator.upper()}**"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "HMM Market Regime", f"{current_regime} {icon}",
            help="BULL = invest 100% | NEUTRAL = invest 50% | BEAR = 100% cash.",
        )
        c2.metric(
            "Capital Exposure", f"{exposure * 100:.0f}%",
            help="Fraction of capital eligible for stocks under the HMM regime.",
        )
        c3.metric(
            "Cash Reserve", f"{(1.0 - exposure) * 100:.0f}%",
            help="Fraction held in cash to protect from a market crash.",
        )
        c4.metric(
            "Universe Size", f"{len(cfg_run.universe)} Stocks",
            help="Number of S&P 500 names successfully ingested for ranking.",
        )

        # ---------------- Model Quality (IC / IR / Shuffled / Walk-forward) ----------------
        # Walk-forward IC is the headline metric: it averages over many regimes
        # and is robust to single-window noise. Shuffled-target IC is the
        # leakage canary — should hover near zero. The single 60d window is
        # kept for backward compatibility but de-emphasized.
        ic_metrics = plan["ic_metrics"]
        if ic_metrics:
            st.divider()
            st.subheader("Model Quality (Out-of-Sample)")

            mean_ic = ic_metrics.get("mean_ic", float("nan"))
            ic_ir = ic_metrics.get("ic_ir", float("nan"))
            window = ic_metrics.get("window", 60)
            shuf_ic = ic_metrics.get("shuffled_ic", float("nan"))
            wf = ic_metrics.get("walkforward") or {}
            wf_mean = wf.get("mean_ic", float("nan"))
            wf_ir = wf.get("ic_ir_ann", float("nan"))
            wf_hit = wf.get("hit_rate", float("nan"))
            wf_days = wf.get("n_days", 0)

            # Quality bands keyed off walk-forward mean IC (the most reliable
            # number on the panel). 0.03+ is tradeable for cross-sectional
            # equity ranking; 0.01–0.03 is marginal. Fall back to the 60d
            # single-window if walk-forward couldn't run.
            if pd.notna(wf_mean) and wf_days > 0:
                quality_basis = "walk-forward"
                if wf_mean > 0.03:
                    quality = "Tradeable"
                elif wf_mean > 0.01:
                    quality = "Marginal"
                elif wf_mean > 0:
                    quality = "Weak"
                else:
                    quality = "Negative"
            elif pd.notna(mean_ic):
                quality_basis = "60d window"
                if mean_ic > 0.03:
                    quality = "Tradeable"
                elif mean_ic > 0.005:
                    quality = "Marginal"
                elif mean_ic > 0:
                    quality = "Weak"
                else:
                    quality = "Negative"
            else:
                quality_basis = "n/a"
                quality = "Insufficient data"

            # Leakage canary — shuffled IC < 0.02 (abs) = features are clean.
            shuf_clean = pd.notna(shuf_ic) and abs(shuf_ic) < 0.02
            shuf_label = "Clean" if shuf_clean else ("Suspicious" if pd.notna(shuf_ic) else "N/A")

            with st.container(border=True):
                # Headline row: signal quality verdict + the three walk-forward
                # numbers that drive it. This is what to look at first.
                q1, q2, q3, q4 = st.columns([1.2, 1, 1, 1])
                q1.metric(
                    "Signal Quality",
                    quality,
                    help=f"Verdict based on {quality_basis} mean IC. "
                         "Tradeable: > 0.03 · Marginal: 0.01–0.03 · "
                         "Weak: 0–0.01 · Negative: ≤ 0.",
                )
                q2.metric(
                    "Today-Screened IC",
                    f"{wf_mean:.4f}" if pd.notna(wf_mean) else "N/A",
                    help="Rolling walk-forward IC on the today-screened top-50 panel. "
                         "Useful as a quick diagnostic, but inflated by liquidity-rank look-ahead. "
                         "Use Diagnostics > Universe Audit for the real IC baseline.",
                )
                q3.metric(
                    "Today-Screened IR",
                    f"{wf_ir:.2f}" if pd.notna(wf_ir) else "N/A",
                    help="Annualized IR of daily ICs on the today-screened panel. "
                         "Use the Universe Audit / Feature Ablation tabs for real rescreened IC.",
                )
                q4.metric(
                    "Hit Rate",
                    f"{wf_hit:.1%}" if pd.notna(wf_hit) else "N/A",
                    help="Fraction of walk-forward test days with IC > 0. "
                         ">55% consistent · 70%+ on today-screened panel = universe look-ahead.",
                )

                st.caption(
                    f"Today-screened walk-forward aggregated **{wf_days:d}** test days · "
                    f"Shuffled-target IC: **{shuf_ic:.4f}** ({shuf_label}) · "
                    f"Single-window IC ({window}d): **{mean_ic:.4f}**"
                    if pd.notna(wf_mean) and pd.notna(shuf_ic) and pd.notna(mean_ic)
                    else "Walk-forward metrics unavailable — see diagnostics below."
                )

                with st.expander("Diagnostics & legacy metrics", expanded=False):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric(
                        f"Single-window IC ({window}d)",
                        f"{mean_ic:.4f}" if pd.notna(mean_ic) else "N/A",
                        help="Most-recent 60d Spearman IC vs realized 5d forward returns. "
                             "Noisy; kept for backward compatibility.",
                    )
                    d2.metric(
                        "Single-window IR",
                        f"{ic_ir:.2f}" if pd.notna(ic_ir) else "N/A",
                        help="Non-annualized IR of the 60d daily IC series (mean/std).",
                    )
                    d3.metric(
                        "Walk-fwd Days",
                        f"{wf_days:d}" if wf_days else "N/A",
                        help="Total test days aggregated. More days = more statistical power.",
                    )
                    d4.metric(
                        "Shuffled-target IC",
                        f"{shuf_ic:.4f}" if pd.notna(shuf_ic) else "N/A",
                        delta=shuf_label, delta_color="off",
                        help="Train on permuted labels, eval on real test set. "
                             "|IC| < 0.02 = features are leak-free.",
                    )
                    st.caption(
                        "**Universe caveat.** ICs above use a **today-screened** "
                        "top-50 panel (tickers selected by TODAY's dollar volume, then 3y "
                        "history loaded). Introduces liquidity-rank look-ahead. For a "
                        "point-in-time estimate that re-screens per walk on the full S&P "
                        "500, open the **🔬 Diagnostics** tab → **Universe Audit**."
                    )

        # ---------------- NEW: HMM Regime Timeline overlay ----------------
        regime_history = plan["regime_history"]
        spy_prices = plan["spy_prices"]
        if (regime_history is not None and not regime_history.empty
                and not spy_prices.empty):
            st.divider()
            st.subheader("HMM Regime Timeline (SPY Overlay)")

            common_idx = spy_prices.index.intersection(regime_history.index)
            spy_subset = spy_prices.loc[common_idx]
            regime_subset = regime_history.loc[common_idx]["Regime"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spy_subset.index, y=spy_subset.values,
                mode='lines', name='SPY',
                line=dict(color='#1f2937', width=2),
            ))

            # Build contiguous regime spans
            spans = []
            if len(regime_subset) > 0:
                cur = regime_subset.iloc[0]
                start = regime_subset.index[0]
                for i in range(1, len(regime_subset)):
                    if regime_subset.iloc[i] != cur:
                        spans.append((start, regime_subset.index[i - 1], cur))
                        cur = regime_subset.iloc[i]
                        start = regime_subset.index[i]
                spans.append((start, regime_subset.index[-1], cur))

            for s_start, s_end, s_regime in spans:
                fig.add_vrect(
                    x0=s_start, x1=s_end,
                    fillcolor=REGIME_FILL.get(s_regime, "rgba(128,128,128,0.10)"),
                    line_width=0, layer="below",
                )

            fig.update_layout(
                title="SPY Price with HMM Regime Bands (Bull · Neutral · Bear)",
                xaxis_title="", yaxis_title="SPY Price ($)",
                hovermode="x unified", showlegend=False, height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            occ = (regime_subset.value_counts(normalize=True)
                   .reindex(["BULL", "NEUTRAL", "BEAR"]).fillna(0))
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Time in BULL", f"{occ['BULL']:.1%}")
            oc2.metric("Time in NEUTRAL", f"{occ['NEUTRAL']:.1%}")
            oc3.metric("Time in BEAR", f"{occ['BEAR']:.1%}")

        # ---------------- Plan body (exposure > 0) ----------------
        if exposure > 0 and len(plan["top_tickers"]) > 0:
            # Hot-swappable active weights from cached HRP/Kelly
            allocator_now = cfg_live.allocator
            active_weights = (plan["hrp_weights"] if allocator_now == "hrp"
                              else plan["kelly_weights"])
            target_weights = active_weights * exposure

            scaled_with_cash = target_weights.copy()
            scaled_with_cash["CASH"] = 1.0 - exposure
            weights_df = scaled_with_cash.reset_index()
            weights_df.columns = ["Ticker", "Target Allocation"]
            display_df = pd.merge(
                plan["scores_df"], weights_df, on="Ticker", how="right"
            ).fillna(0).sort_values(by="Target Allocation", ascending=False)

            st.divider()
            st.subheader("Projected Execution Plan")
            st.caption(
                "**Predicted Rank:** XGBoost confidence (higher = more bullish). "
                "**Target Allocation:** post-exposure portfolio slice."
            )

            ct, cc = st.columns([1, 1])
            with ct:
                st.dataframe(
                    display_df.style.format({
                        "Predicted Rank": "{:.2%}",
                        "Target Allocation": "{:.2%}",
                    }),
                    width="stretch", hide_index=True,
                )
            with cc:
                pie_data = display_df[display_df["Target Allocation"] > 0]
                fig_pie = px.pie(
                    pie_data, values='Target Allocation', names='Ticker',
                    title=f'{allocator_now.upper()} Portfolio Allocation', hole=0.4,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # ---------------- NEW: Sector Distribution ----------------
            sector_map = plan["sector_map"]
            top_tickers = plan["top_tickers"]
            if sector_map and top_tickers:
                st.divider()
                st.subheader("Sector Distribution")

                sector_rows = []
                for t in top_tickers:
                    sector_rows.append({
                        "Sector": sector_map.get(t, "Unknown"),
                        "Ticker": t,
                        "Weight": float(active_weights.get(t, 0)),
                    })
                sec_df = pd.DataFrame(sector_rows)
                sec_agg = (sec_df.groupby("Sector")["Weight"].sum()
                           .reset_index().sort_values("Weight", ascending=False))

                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    fig_sec = px.pie(
                        sec_agg, values="Weight", names="Sector",
                        title=f"Top-{len(top_tickers)} Sector Concentration", hole=0.3,
                    )
                    st.plotly_chart(fig_sec, use_container_width=True)
                with sc2:
                    fig_bar = px.bar(
                        sec_df.sort_values("Weight", ascending=True),
                        x="Weight", y="Ticker", color="Sector",
                        orientation="h", title="Weight by Ticker (Colored by Sector)",
                    )
                    fig_bar.update_layout(xaxis_tickformat=".0%")
                    st.plotly_chart(fig_bar, use_container_width=True)

                n_secs = sec_df["Sector"].nunique()
                st.caption(
                    f"Sector cap: max **{cfg_run.max_names_per_sector}** names per GICS sector. "
                    f"Picks use **{n_secs}** sectors across **{len(top_tickers)}** tickers."
                )

            # ---------------- NEW: Allocator Comparison ----------------
            hrp_w = plan["hrp_weights"]
            kelly_w = plan["kelly_weights"]
            if not hrp_w.empty and not kelly_w.empty:
                st.divider()
                st.subheader("Allocator Comparison (HRP vs Kelly)")
                st.caption(
                    f"Currently using: **{allocator_now.upper()}** — change in sidebar to hot-swap "
                    "without re-running the pipeline."
                )

                comp = []
                for t in sorted(set(hrp_w.index) | set(kelly_w.index)):
                    comp.append({"Ticker": t, "Allocator": "HRP",
                                 "Weight": float(hrp_w.get(t, 0))})
                    comp.append({"Ticker": t, "Allocator": "Kelly",
                                 "Weight": float(kelly_w.get(t, 0))})
                comp_df = pd.DataFrame(comp)

                fig_cmp = px.bar(
                    comp_df, x="Ticker", y="Weight", color="Allocator",
                    barmode="group", title="HRP vs Kelly Weight Allocation",
                    color_discrete_map=ALLOC_COLOR,
                )
                fig_cmp.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig_cmp, use_container_width=True)

                hhi_hrp = float((hrp_w ** 2).sum())
                hhi_kly = float((kelly_w ** 2).sum())
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("HRP Max Weight", f"{hrp_w.max():.1%}")
                cm2.metric("Kelly Max Weight", f"{kelly_w.max():.1%}")
                cm3.metric(
                    "HRP HHI", f"{hhi_hrp:.3f}",
                    help="Herfindahl-Hirschman Index. Lower = more diversified (min = 1/N).",
                )
                cm4.metric("Kelly HHI", f"{hhi_kly:.3f}")

            # ---------------- NEW: Slicing Schedule Preview ----------------
            latest_prices = plan["latest_prices"]
            if cfg_live.slicing_enabled and not target_weights.empty:
                st.divider()
                st.subheader("Order Slicing Preview")

                # Read live equity for accurate sizing; fall back to $100k
                est_equity = 100_000.0
                try:
                    if api_key and api_secret:
                        client_tmp = TradingClient(api_key, api_secret, paper=is_paper)
                        est_equity = float(client_tmp.get_account().equity)
                except Exception:
                    pass

                positive = target_weights[target_weights > 0].sort_values(ascending=False)
                if not positive.empty and positive.index[0] in latest_prices.index:
                    top_sym = positive.index[0]
                    top_w = float(positive.iloc[0])
                    price = float(latest_prices.loc[top_sym])
                    qty = round((top_w * est_equity) / price, 4)
                    notional = qty * price
                    threshold_notional = cfg_live.slicing_threshold_pct * est_equity
                    would_slice = notional > threshold_notional and qty > 0

                    with st.container(border=True):
                        sp1, sp2, sp3 = st.columns(3)
                        sp1.metric(
                            "Top Buy", top_sym,
                            help=f"Largest planned buy by weight ({top_w:.1%}).",
                        )
                        sp2.metric(
                            "Qty / Notional",
                            f"{qty:,.2f} sh",
                            f"${notional:,.0f}",
                            help="At assumed equity & current price.",
                            delta_color="off",
                        )
                        sp3.metric(
                            "Would Slice?", "Yes" if would_slice else "No",
                            help=f"Slice threshold: ${threshold_notional:,.0f} "
                                 f"({cfg_live.slicing_threshold_pct:.0%} of equity).",
                        )

                        if would_slice:
                            schedule = compute_slice_schedule(
                                qty, cfg_live.slicing_num_children, cfg_live.slicing_convexity
                            )
                            sched_df = pd.DataFrame({
                                "Child": [f"#{i+1}" for i in range(len(schedule))],
                                "Qty": schedule,
                                "Notional ($)": [round(q * price, 2) for q in schedule],
                                "% of Parent": [q / qty if qty > 0 else 0 for q in schedule],
                            })
                            fig_sl = px.bar(
                                sched_df, x="Child", y="Qty",
                                title=f"Convex front-loaded schedule for {top_sym} "
                                      f"(convexity={cfg_live.slicing_convexity}, "
                                      f"{len(schedule)} children, "
                                      f"~{cfg_live.slicing_window_minutes} min window)",
                                color_discrete_sequence=["#3b82f6"],
                            )
                            fig_sl.update_layout(showlegend=False, yaxis_title="Shares")
                            st.plotly_chart(fig_sl, use_container_width=True)

                            st.dataframe(
                                sched_df.style.format({
                                    "Qty": "{:,.4f}",
                                    "Notional ($)": "${:,.2f}",
                                    "% of Parent": "{:.1%}",
                                }),
                                width="stretch", hide_index=True,
                            )
                            st.caption(
                                f"Each child reuses the parent limit price. "
                                f"Convexity > 1 front-loads execution (more shares early)."
                            )
                        else:
                            st.info(
                                f"Order would be submitted as a single transaction "
                                f"(notional ${notional:,.0f} ≤ threshold ${threshold_notional:,.0f})."
                            )
                else:
                    st.info("No buys in current plan to slice.")

            # ---------------- NEW: Execute Rebalance ----------------
            st.divider()
            st.subheader("Execute Rebalance")
            with st.container(border=True):
                exec_help_live = (
                    "Submit real orders to Alpaca "
                    + ("(**PAPER** account)" if is_paper else "(**LIVE MONEY**)")
                )
                st.markdown(
                    f"**Account mode:** {'PAPER' if is_paper else 'LIVE'} · "
                    f"**Allocator:** {allocator_now.upper()} · "
                    f"**Slicing:** {'ON' if cfg_live.slicing_enabled else 'OFF'} · "
                    f"**Breaker:** {'ON' if cfg_live.breaker_enabled else 'OFF'}"
                )

                ex1, ex2 = st.columns(2)
                run_dry = ex1.button(
                    "Execute Dry-Run", type="secondary", use_container_width=True,
                    help="Compute and log what would be done without submitting orders.",
                )

                with ex2:
                    confirm_live = st.checkbox(
                        "I confirm live execution",
                        key="confirm_live_exec",
                        disabled=run_dry,
                    )
                    run_live = st.button(
                        "Execute LIVE",
                        type="primary", use_container_width=True,
                        disabled=not confirm_live,
                        help=exec_help_live,
                    )

                if run_dry or run_live:
                    is_dry = bool(run_dry)
                    label = "DRY-RUN" if is_dry else "LIVE"

                    # Refresh cfg with current sidebar at execute time (allocator/breaker/slicing).
                    # Keep universe from plan since re-scraping would be wasteful and inconsistent.
                    cfg_exec = make_config()
                    cfg_exec.universe = cfg_run.universe

                    with st.spinner(f"Executing {label} rebalance — this may take up to a minute..."):
                        buf = io.StringIO()
                        success = True
                        err_msg = ""
                        try:
                            with contextlib.redirect_stdout(buf):
                                engine = ExecutionEngine(cfg_exec)
                                gz = DrawdownController(
                                    cfg_exec.drawdown_floor_alpha,
                                    cfg_exec.drawdown_leverage_k,
                                    cfg_exec.gz_state_path,
                                )
                                engine.attach_gz_controller(gz)
                                engine.send_rebalance_orders(
                                    target_weights, latest_prices, dry_run=is_dry,
                                )
                        except Exception as e:
                            success = False
                            err_msg = str(e)

                    output = buf.getvalue()
                    if success:
                        st.success(f"{label} execution complete.")
                    else:
                        st.error(f"{label} execution failed: {err_msg}")

                    if output:
                        with st.expander("Engine Output", expanded=True):
                            st.code(output, language="text")

            # ---------------- Export Report ----------------
            st.divider()
            st.subheader("Export Run Report")
            st.caption("Hover the box and click 'Copy' in the top right to paste to your AI.")
            report = "**PHASE 3 PIPELINE RUN**\n"
            report += f"- **Regime:** {current_regime}\n"
            report += f"- **Exposure:** {exposure * 100:.0f}%\n"
            report += f"- **Universe Size:** {len(cfg_run.universe)} Stocks\n"
            report += f"- **Allocator:** {allocator_now.upper()}\n"
            if ic_metrics and pd.notna(ic_metrics.get("mean_ic", float("nan"))):
                report += (
                    f"- **OOS Rank IC ({ic_metrics['window']}d):** "
                    f"{ic_metrics['mean_ic']:.4f} | "
                    f"IC IR: {ic_metrics['ic_ir']:.2f}\n"
                )
            if ic_metrics:
                shuf_ic_r = ic_metrics.get("shuffled_ic", float("nan"))
                if pd.notna(shuf_ic_r):
                    canary = "clean" if abs(shuf_ic_r) < 0.02 else "suspicious"
                    report += f"- **Shuffled-target IC:** {shuf_ic_r:.4f} ({canary})\n"
                wf_r = ic_metrics.get("walkforward") or {}
                if wf_r.get("n_days", 0):
                    report += (
                        f"- **Today-screened walk-forward IC:** mean {wf_r['mean_ic']:.4f} | "
                        f"IR(ann) {wf_r['ic_ir_ann']:.2f} | "
                        f"hit {wf_r['hit_rate']:.1%} | "
                        f"{wf_r['n_days']} test days across {wf_r['n_walks']} walks "
                        f"(today-screened panel — see audit caveat)\n"
                    )
            report += "\n**Target Allocations:**\n"
            report += display_df.to_markdown(index=False)
            st.code(report, language="markdown")

        elif exposure == 0:
            st.divider()
            st.error("Market is currently in a BEAR Regime. XGBoost bypassed. System holds 100% Cash.")

# =============================================================================
# TAB 3: DIAGNOSTICS (cMDA + Universe Audit)
# =============================================================================
with tab_diagnostics:
    sub_cmda, sub_audit, sub_ablation, sub_attrib = st.tabs([
        "Feature Importance (cMDA)",
        "Universe Audit",
        "Feature Ablation",
        "📊 Performance Attribution",
    ])

    # ---------------------------------------------------------- cMDA sub-tab
    with sub_cmda:
        st.markdown(
            "Run a **Cluster-Based Mean Decrease Accuracy** diagnostic on the current "
            "feature set. Features are grouped by correlation-distance hierarchical "
            "clustering, then each cluster is jointly shuffled on a held-out validation "
            "window to measure the drop in Spearman IC. Higher drop = more useful cluster. "
            "This is **observational only** — no trades are placed and the live training "
            "pipeline is not affected."
        )

        c1, c2 = st.columns(2)
        cmda_threshold = c1.slider(
            "Cluster distance threshold", min_value=0.10, max_value=1.00, value=0.50, step=0.05,
            help="Cut height on the dendrogram. Lower = more, tighter clusters; higher = fewer, broader.",
        )
        cmda_perms = c2.slider(
            "Permutations per cluster", min_value=5, max_value=50, value=10, step=5,
            help="More permutations stabilize the estimate at the cost of runtime.",
        )

        run_cmda = st.button("Run cMDA Diagnostic", type="primary", key="run_cmda_btn")

        if run_cmda:
            cfg = make_config()

            with st.spinner(f"Scanning S&P 500 for the top {cfg.universe_size} most liquid stocks..."):
                fetcher = DataFetcher()
                dynamic_tickers, sector_map = fetcher.get_dynamic_universe(top_n=cfg.universe_size)
                cfg.universe = tuple(dynamic_tickers)

            with st.spinner("📥 Fetching 3 years of market data..."):
                ohlc = fetcher.fetch_daily_ohlc(list(cfg.universe) + ["SPY"], period="3y")
                prices = ohlc["Close"]
                opens = ohlc["Open"]

            with st.spinner("🧪 Engineering features..."):
                signals = MLSignalGenerator(cfg, prices, opens, sector_map=sector_map)
                dataset = signals._engineer_features(prices)

            with st.spinner(f"Running cMDA ({cmda_perms} permutations × clusters at d≤{cmda_threshold})..."):
                cmda_df = signals.run_cmda_diagnostic(
                    dataset, dist_threshold=float(cmda_threshold),
                    n_permutations=int(cmda_perms),
                )

            st.divider()
            st.subheader("Feature-Cluster Importance Ranking")
            st.caption(
                "**mean_importance** = baseline IC minus permuted IC (averaged over runs). "
                "Positive = removing this cluster hurts predictions; near-zero = redundant or noise."
            )

            display_df = cmda_df.copy()
            display_df["features_in_cluster"] = display_df["features_in_cluster"].apply(", ".join)
            st.dataframe(
                display_df.style.format({
                    "mean_importance": "{:+.4f}",
                    "std_importance": "{:.4f}",
                }),
                width="stretch", hide_index=True,
            )

            fig_bar = px.bar(
                display_df, x="features_in_cluster", y="mean_importance",
                error_y="std_importance",
                title="Mean Decrease in Spearman IC by Cluster (±1σ)",
                labels={"features_in_cluster": "Cluster Features", "mean_importance": "ΔIC"},
            )
            fig_bar.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()
            st.subheader("Export Results")
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            json_bytes = cmda_df.to_json(orient="records", indent=2).encode("utf-8")

            d1, d2 = st.columns(2)
            d1.download_button(
                "Download CSV", data=csv_bytes,
                file_name=f"cmda_results_{ts}.csv", mime="text/csv",
                width="stretch",
            )
            d2.download_button(
                "Download JSON", data=json_bytes,
                file_name=f"cmda_results_{ts}.json", mime="application/json",
                width="stretch",
            )

            st.caption("Or copy the markdown report below to share with an AI:")
            report = "**cMDA FEATURE-CLUSTER IMPORTANCE**\n"
            report += f"- **Universe Size:** {len(cfg.universe)} stocks\n"
            report += f"- **Distance threshold:** {cmda_threshold:.2f}\n"
            report += f"- **Permutations per cluster:** {cmda_perms}\n"
            report += f"- **Clusters formed:** {len(cmda_df)}\n\n"
            report += display_df.to_markdown(index=False)
            st.code(report, language="markdown")

    # ----------------------------------------------------- Universe Audit sub-tab
    with sub_audit:
        st.markdown(
            "Re-runs the walk-forward IC over the **full current S&P 500** with "
            "per-walk top-N rescreening by training-window dollar volume. This "
            "controls for the liquidity-rank look-ahead in the default `Run Manual "
            "AI Calculation` panel (which screens by **today's** dollar volume). "
            "**Observational only** — no trades are placed."
        )
        st.caption(
            "Heavy operation. Downloads 3y of OHLCV for ~500 tickers from "
            "yfinance, then runs walk-forward. Typically 1–3 minutes; longer on "
            "first run when the data isn't cached."
        )

        run_audit = st.button("Run Universe Audit", type="primary", key="run_audit_btn")

        if run_audit:
            buf = io.StringIO()
            success = True
            err_msg = ""
            wf_result = None
            with st.spinner("Auditing universe (pulling S&P 500 list + 3y OHLCV + walk-forward)…"):
                try:
                    with contextlib.redirect_stdout(buf):
                        wf_result = run_universe_audit()
                except Exception as e:
                    success = False
                    err_msg = str(e)

            if success and wf_result:
                st.session_state.audit_universe_result = {
                    "wf": wf_result,
                    "stdout": buf.getvalue(),
                    "computed_at": pd.Timestamp.now(),
                }
            elif not success:
                st.error(f"Audit failed: {err_msg}")
                if buf.getvalue():
                    with st.expander("Partial output", expanded=True):
                        st.code(buf.getvalue(), language="text")

        # Render cached audit result (survives Streamlit reruns)
        if "audit_universe_result" in st.session_state:
            cached = st.session_state.audit_universe_result
            wf_r = cached["wf"]

            st.divider()
            st.caption(
                f"Audit computed at **{cached['computed_at'].strftime('%Y-%m-%d %H:%M:%S')}**. "
                f"Compare to today-screened walk-forward mean IC ≈ 0.106, IR(ann) ≈ 7.65."
            )

            with st.container(border=True):
                a1, a2, a3, a4 = st.columns(4)
                a1.metric(
                    "Mean Daily IC", f"{wf_r.get('mean_ic', float('nan')):.4f}",
                    help="True out-of-sample mean IC under per-walk rescreening. "
                         "≈ 0.028–0.045 = realistic for cross-sectional equity ranking.",
                )
                a2.metric(
                    "IR (annualized)", f"{wf_r.get('ic_ir_ann', float('nan')):.2f}",
                    help=">2 = strong, >1 = good, >0.5 = acceptable.",
                )
                a3.metric(
                    "Hit Rate", f"{wf_r.get('hit_rate', float('nan')):.1%}",
                    help="Fraction of test days with IC > 0.",
                )
                a4.metric(
                    "Std Daily IC", f"{wf_r.get('std_ic', float('nan')):.4f}",
                    help="Day-to-day IC volatility across walks.",
                )

                b1, b2, b3 = st.columns(3)
                b1.metric("Walks", f"{wf_r.get('n_walks', 0):d}")
                b2.metric("Test Days", f"{wf_r.get('n_days', 0):d}")
                jaccard = wf_r.get("avg_walk_to_walk_universe_overlap", float("nan"))
                b3.metric(
                    "Universe Stability (Jaccard)",
                    f"{jaccard:.1%}" if pd.notna(jaccard) else "N/A",
                    help="Avg walk-to-walk overlap of the screened top-N universe. "
                         "Higher = picks rotate less between adjacent walks.",
                )

            with st.expander("Full audit output", expanded=False):
                st.code(cached["stdout"], language="text")

            st.subheader("Copy Report")
            audit_report = "**UNIVERSE AUDIT — broad S&P 500 + per-walk rescreening**\n"
            audit_report += f"- **Mean IC:** {wf_r.get('mean_ic', float('nan')):.4f}\n"
            audit_report += f"- **Std IC:** {wf_r.get('std_ic', float('nan')):.4f}\n"
            audit_report += f"- **IR (ann):** {wf_r.get('ic_ir_ann', float('nan')):.2f}\n"
            audit_report += f"- **Hit Rate:** {wf_r.get('hit_rate', float('nan')):.1%}\n"
            audit_report += f"- **Walks / Test Days:** {wf_r.get('n_walks', 0)} / {wf_r.get('n_days', 0)}\n"
            audit_report += f"- **Universe Jaccard:** {jaccard:.1%}\n" if pd.notna(jaccard) else ""
            st.code(audit_report, language="markdown")

    # ------------------------------------------------- Feature Ablation sub-tab
    with sub_ablation:
        st.markdown(
            "Compares feature sets using the **same broad S&P 500 per-walk "
            "rescreening** as the Universe Audit. This is the real-IC test path; "
            "the normal manual run diagnostics are still today-screened and can "
            "look inflated. **Observational only** - no trades are placed."
        )
        st.caption(
            "Tests: production full set, momentum-only, momentum+volatility, "
            "and volatility-only. The live ranker uses `PRODUCTION_FEATURES`, "
            "currently the full feature set because it won the real rescreened audit."
        )

        run_ablation = st.button(
            "Run Feature Ablation Audit", type="primary", key="run_feature_ablation_btn"
        )

        if run_ablation:
            buf = io.StringIO()
            success = True
            err_msg = ""
            ablation_result = None
            with st.spinner("Running feature ablation on broad rescreened universe..."):
                try:
                    with contextlib.redirect_stdout(buf):
                        ablation_result = run_feature_ablation_audit()
                except Exception as e:
                    success = False
                    err_msg = str(e)

            if success and ablation_result:
                st.session_state.feature_ablation_result = {
                    "result": ablation_result,
                    "stdout": buf.getvalue(),
                    "computed_at": pd.Timestamp.now(),
                }
            elif not success:
                st.error(f"Feature ablation failed: {err_msg}")
                if buf.getvalue():
                    with st.expander("Partial output", expanded=True):
                        st.code(buf.getvalue(), language="text")

        if "feature_ablation_result" in st.session_state:
            cached = st.session_state.feature_ablation_result
            result = cached["result"]
            summary = result["summary"].copy()

            st.divider()
            st.caption(
                f"Ablation computed at **{cached['computed_at'].strftime('%Y-%m-%d %H:%M:%S')}**. "
                "Sorted by real rescreened mean IC."
            )

            display_summary = summary.copy()
            display_summary["hit_rate"] = display_summary["hit_rate"].map(lambda x: f"{x:.1%}")
            display_summary["avg_walk_to_walk_universe_overlap"] = display_summary[
                "avg_walk_to_walk_universe_overlap"
            ].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            st.dataframe(
                display_summary.style.format({
                    "mean_ic": "{:.4f}",
                    "std_ic": "{:.4f}",
                    "ic_ir_ann": "{:.2f}",
                }),
                width="stretch",
            )

            best = summary.iloc[0]
            current = summary[summary["feature_set"] == "production_full"]
            current_ic = (
                float(current.iloc[0]["mean_ic"]) if len(current) else float("nan")
            )
            st.info(
                f"Best feature set: **{best['feature_set']}** "
                f"(IC {best['mean_ic']:.4f}, IR {best['ic_ir_ann']:.2f}). "
                f"Production set IC: **{current_ic:.4f}**."
            )

            with st.expander("Full ablation output", expanded=False):
                st.code(cached["stdout"], language="text")

            report = "**FEATURE ABLATION AUDIT - broad S&P 500 + per-walk rescreening**\n"
            report += f"- **Production features:** {', '.join(PRODUCTION_FEATURES)}\n\n"
            report += summary.to_markdown(index=False)
            st.code(report, language="markdown")

    # ----------------------------------------------------- Attribution Backtest sub-tab
    with sub_attrib:
        st.markdown(
            "Run a walk-forward, long-only backtest of the top-N ML strategy. Decomposes "
            "realized returns into market beta (SPY), sector tilts, and residual "
            "stock-selection alpha via regression. This isolates how much P&L comes from "
            "market timing vs. true stock-picking skill."
        )

        if "plan" not in st.session_state or "full_prices" not in st.session_state.plan:
            st.info("👆 Please run the **Manual AI Calculation** in the Research tab first to populate the session data.")
        else:
            attrib_allocator = st.radio(
                "Allocation Method for Backtest", ["HRP", "Equal-Weight"], horizontal=True,
                help="Choose the allocation strategy for this attribution backtest."
            )
            st.markdown("**Live-bot overlays**")
            c1, c2, c3 = st.columns(3)
            with c1:
                attrib_use_hmm = st.checkbox(
                    "HMM regime exposure", value=True,
                    help="Scale each rebalance by the historical SPY HMM regime exposure."
                )
                attrib_use_sector_cap = st.checkbox(
                    "Sector cap", value=True,
                    help="Use the live max-names-per-sector basket selection rule."
                )
            with c2:
                attrib_use_gz = st.checkbox(
                    "GZ drawdown floor", value=True,
                    help="Simulate the Grossman-Zhou high-water-mark exposure overlay."
                )
                attrib_use_sector_neutral = st.checkbox(
                    "Sector-neutral signal", value=True,
                    help="Residualize ML predictions by GICS sector before ranking."
                )
            with c3:
                attrib_use_exit_discipline = st.checkbox(
                    "Exit discipline", value=True,
                    help="Apply time-stop and stop-loss exits inside each holding window."
                )
                attrib_rebalance_freq = st.number_input(
                    "Rebalance days", min_value=1, max_value=30, value=5, step=1,
                    help="Holding window between attribution rebalances."
                )

            if attrib_use_sector_cap:
                cfg_preview = make_config()
                st.caption(
                    f"Sector cap uses max {cfg_preview.max_names_per_sector} names per sector. "
                    f"Exit discipline uses {cfg_preview.time_stop_days} day time-stop and "
                    f"{cfg_preview.stop_loss_pct:.0%} stop-loss."
                )
            run_attrib = st.button("Run Attribution Backtest", type="primary", key="run_attrib_btn")

            if run_attrib:
                plan = st.session_state.plan
                full_prices = plan.get("full_prices")
                sector_map = plan.get("sector_map")
                cfg = make_config()

                buf = io.StringIO()
                success = True
                err_msg = ""
                attrib_result = None
                with st.spinner("Running attribution backtest (walk-forward training, holding, and regression)..."):
                    try:
                        with contextlib.redirect_stdout(buf):
                            fetcher = DataFetcher()
                            ohlc = fetcher.fetch_daily_ohlc(list(full_prices.columns), period="3y")
                            opens = ohlc["Open"]
                            
                            backtester = AttributionBacktest()
                            attrib_result = backtester.run(
                                full_prices, opens, sector_map, cfg,
                                rebalance_freq=int(attrib_rebalance_freq),
                                allocation_method=attrib_allocator.split('-')[0].lower(),
                                use_hmm=bool(attrib_use_hmm),
                                use_gz=bool(attrib_use_gz),
                                use_exit_discipline=bool(attrib_use_exit_discipline),
                                use_sector_cap=bool(attrib_use_sector_cap),
                                use_sector_neutral_signal=bool(attrib_use_sector_neutral),
                            )
                    except Exception as e:
                        success = False
                        err_msg = str(e)
                        buf.write("\n\nTRACEBACK:\n")
                        buf.write(traceback.format_exc())

                if success and attrib_result:
                    st.session_state.attribution_result = {
                        "result": attrib_result,
                        "stdout": buf.getvalue(),
                        "computed_at": pd.Timestamp.now(),
                        "settings": {
                            "allocator": attrib_allocator,
                            "rebalance_freq": int(attrib_rebalance_freq),
                            "hmm": bool(attrib_use_hmm),
                            "gz": bool(attrib_use_gz),
                            "exit_discipline": bool(attrib_use_exit_discipline),
                            "sector_cap": bool(attrib_use_sector_cap),
                            "sector_neutral_signal": bool(attrib_use_sector_neutral),
                        },
                    }
                elif not success:
                    st.error(f"Attribution backtest failed: {err_msg}")
                    if buf.getvalue():
                        with st.expander("Partial output", expanded=True):
                            st.code(buf.getvalue(), language="text")
            
            # Render cached result
            if "attribution_result" in st.session_state:
                cached = st.session_state.attribution_result
                res = cached["result"]
                metrics = res["metrics"]
                data = res["data"]
                
                st.divider()
                st.caption(
                    f"Backtest computed at **{cached['computed_at'].strftime('%Y-%m-%d %H:%M:%S')}**."
                )
                settings = cached.get("settings", {})
                if settings:
                    enabled = [
                        label for key, label in [
                            ("hmm", "HMM"),
                            ("gz", "GZ"),
                            ("exit_discipline", "Exit discipline"),
                            ("sector_cap", "Sector cap"),
                            ("sector_neutral_signal", "Sector-neutral signal"),
                        ]
                        if settings.get(key)
                    ]
                    st.caption(
                        f"Allocator: **{settings.get('allocator', 'n/a')}** | "
                        f"Rebalance: **{settings.get('rebalance_freq', 'n/a')}d** | "
                        f"Overlays: **{', '.join(enabled) if enabled else 'None'}**"
                    )

                st.subheader("Backtest Equity Curve")
                equity_curve = data["equity_curve"]
                fig_eq = px.line(equity_curve, title="Strategy Equity Curve")
                fig_eq.update_layout(yaxis_title="Cumulative Return", xaxis_title="", showlegend=False)
                st.plotly_chart(fig_eq, use_container_width=True)

                st.subheader("Performance & Attribution Metrics")
                with st.container(border=True):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Annualized Alpha", f"{metrics['annualized_alpha_pct']:.2f}%")
                    m2.metric("SPY Beta", f"{metrics['spy_beta']:.3f}")
                    m3.metric("R-Squared", f"{metrics['r_squared']:.3f}")
                    m4.metric("Annualized Sharpe", f"{metrics['annualized_sharpe']:.2f}")
                    m5.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

                st.caption(
                    "**Interpretation:** Near-zero residual alpha combined with a high R-Squared (>0.5) suggests that "
                    "most of the strategy's returns are explained by exposure to market beta (SPY) and sector tilts, "
                    "rather than from stock-selection skill ('alpha')."
                )

                exposures = data.get("exposures", pd.DataFrame())
                if isinstance(exposures, pd.DataFrame) and not exposures.empty:
                    st.subheader("Live Overlay Exposure Path")
                    summary_cols = st.columns(4)
                    if "selected_names" in exposures.columns:
                        summary_cols[0].metric("Avg Selected Names", f"{exposures['selected_names'].mean():.1f}")
                    if "selected_sectors" in exposures.columns:
                        summary_cols[1].metric("Avg Active Sectors", f"{exposures['selected_sectors'].mean():.1f}")
                    if "open_positions" in exposures.columns:
                        summary_cols[2].metric("Avg Open Positions", f"{exposures['open_positions'].mean():.1f}")
                    if "forced_exits" in exposures.columns:
                        summary_cols[3].metric("Forced Exits / Rebal", f"{exposures['forced_exits'].mean():.2f}")

                    exp_cols = [c for c in ["hmm_exposure", "gz_exposure", "final_exposure"] if c in exposures.columns]
                    fig_exp = px.line(
                        exposures[exp_cols],
                        title="HMM / GZ Exposure Multipliers by Rebalance",
                    )
                    fig_exp.update_layout(yaxis_title="Exposure", xaxis_title="")
                    st.plotly_chart(fig_exp, use_container_width=True)
                    with st.expander("Exposure ledger", expanded=False):
                        st.dataframe(exposures, use_container_width=True)

                exit_events = data.get("exit_events", pd.DataFrame())
                if isinstance(exit_events, pd.DataFrame) and not exit_events.empty:
                    st.subheader("Exit Discipline Events")
                    e1, e2 = st.columns(2)
                    e1.metric("Forced Exits", f"{len(exit_events):,}")
                    e2.metric("Avg Exit Return", f"{exit_events['return_pct'].mean():.2f}%")
                    with st.expander("Exit event ledger", expanded=False):
                        st.dataframe(exit_events, use_container_width=True)

                st.divider()
                st.subheader("Copy Report")
                report = "**ATTRIBUTION BACKTEST REPORT**\n"
                if settings:
                    enabled = [
                        label for key, label in [
                            ("hmm", "HMM"),
                            ("gz", "GZ"),
                            ("exit_discipline", "Exit discipline"),
                            ("sector_cap", "Sector cap"),
                            ("sector_neutral_signal", "Sector-neutral signal"),
                        ]
                        if settings.get(key)
                    ]
                    report += f"- **Allocator:** {settings.get('allocator', 'n/a')}\n"
                    report += f"- **Rebalance Days:** {settings.get('rebalance_freq', 'n/a')}\n"
                    report += f"- **Live Overlays:** {', '.join(enabled) if enabled else 'None'}\n"
                report += f"- **Annualized Alpha:** {metrics['annualized_alpha_pct']:.2f}%\n"
                report += f"- **SPY Beta:** {metrics['spy_beta']:.3f}\n"
                report += f"- **R-Squared:** {metrics['r_squared']:.3f}\n"
                report += f"- **Annualized Sharpe:** {metrics['annualized_sharpe']:.2f}\n"
                report += f"- **Max Drawdown:** {metrics['max_drawdown_pct']:.2f}%\n\n"
                if isinstance(exposures, pd.DataFrame) and not exposures.empty:
                    if "selected_names" in exposures.columns:
                        report += f"- **Avg Selected Names:** {exposures['selected_names'].mean():.1f}\n"
                    if "selected_sectors" in exposures.columns:
                        report += f"- **Avg Active Sectors:** {exposures['selected_sectors'].mean():.1f}\n"
                    if "open_positions" in exposures.columns:
                        report += f"- **Avg Open Positions:** {exposures['open_positions'].mean():.1f}\n"
                    if "forced_exits" in exposures.columns:
                        report += f"- **Forced Exits / Rebalance:** {exposures['forced_exits'].mean():.2f}\n"
                    report += "\n"
                report += "**Sector Betas:**\n"
                sector_betas = pd.Series(res["sector_betas"]).sort_values(ascending=False)
                report += sector_betas.to_string()
                st.code(report, language="markdown")

                with st.expander("Full backtest output", expanded=False):
                    st.code(cached["stdout"], language="text")

# =============================================================================
# TAB 4: SYSTEM AUDIT ENGINE
# =============================================================================
with tab_system_audit:
    st.markdown(
        "Run the complete 7-component diagnostic engine to empirically isolate value creation "
        "and destruction across your entire quantitative pipeline."
    )
    
    if "plan" not in st.session_state or "full_prices" not in st.session_state.plan:
        st.info("👆 Please run the **Manual AI Calculation** in the Research tab first to populate the session data.")
    else:
        plan = st.session_state.plan
        full_prices = plan.get("full_prices", pd.DataFrame())
        top_tickers = plan.get("top_tickers", [])
        regime_history = plan.get("regime_history", pd.DataFrame())
        
        run_full_audit = st.button("Execute Full Pipeline Audit", type="primary", use_container_width=True)
        
        if run_full_audit and not full_prices.empty:
            cfg = make_config()
            
            # --- PROGRESS BAR UI ---
            progress_text = "Initializing Audit Engine..."
            audit_bar = st.progress(0, text=progress_text)
            
            # --- DATA PREP ---
            audit_bar.progress(10, text="Fetching Open prices for decomposition...")
            fetcher = DataFetcher()
            ohlc = fetcher.fetch_daily_ohlc(list(full_prices.columns), period="3y")
            opens = ohlc["Open"]
            
            audit_bar.progress(20, text="Engineering feature dataset for signal analysis...")
            signals = MLSignalGenerator(cfg, full_prices, opens)
            dataset = signals._engineer_features(full_prices)
            
            # --- TRAIN OOS MODEL FOR DIAGNOSTICS 3 & 5 ---
            audit_bar.progress(25, text="Training diagnostic XGBoost model...")
            feature_cols = PRODUCTION_FEATURES
            
            # Drop rows where target or all features are NA
            dataset_clean = dataset.dropna(subset=['target_rank'] + feature_cols, how='any')
            
            # Time-based split
            unique_dates = sorted(dataset_clean.index.get_level_values('Date').unique())
            train_cutoff_idx = int(len(unique_dates) * 0.7)
            train_cutoff_date = unique_dates[train_cutoff_idx]
            
            train_df = dataset_clean[dataset_clean.index.get_level_values('Date') <= train_cutoff_date]
            test_df = dataset_clean[dataset_clean.index.get_level_values('Date') > train_cutoff_date]

            X_train, y_train = train_df[feature_cols], train_df["target_rank"]
            X_test = test_df[feature_cols]

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            model.fit(X_train, y_train)
            
            # Add predictions to a copy of the test set
            dataset_oos = test_df.copy()
            dataset_oos["_pred"] = model.predict(X_test)
            
            st.caption(
                f"Diagnostic dataset: {len(dataset_clean):,} rows across "
                f"{len(unique_dates)} dates · train ≤ {train_cutoff_date.date()} "
                f"({len(train_df):,} rows) · test > {train_cutoff_date.date()} "
                f"({len(test_df):,} rows)"
            )
            
            # --- EXECUTE 7 COMPONENTS ---
            audit_bar.progress(30, text="Comp 1: Auditing Feature Integrity...")
            f_audit = FeatureIntegrityAudit(full_prices, opens)
            res_comp1 = f_audit.detect_lookahead_bias(["ret_1d"])
            
            audit_bar.progress(40, text="Comp 2: Validating Target Labels...")
            t_audit = TargetLabelValidator(full_prices)
            res_comp2 = t_audit.validate_forward_return_calculation()
            
            audit_bar.progress(50, text="Comp 3: Analyzing IC Translation...")
            res_comp3 = ICtoReturnsTranslator.compute_ic_return_decomposition(dataset_oos)
                
            audit_bar.progress(60, text="Comp 4: Stress-Testing Allocators...")
            a_audit = AllocatorStressTest()
            res_comp4 = a_audit.compare_allocators_on_history(full_prices, top_tickers, cfg)
            
            audit_bar.progress(70, text="Comp 5: Computing Alpha Decay...")
            h_audit = HoldingPeriodAnalysis()
            # dataset_oos already carries the held-out '_pred' column; pass it directly.
            res_comp5 = h_audit.compute_alpha_decay_curve(dataset_oos)
            
            audit_bar.progress(80, text="Comp 6: Simulating Execution Friction...")
            e_audit = ExecutionFrictionAudit()
            allocator_now = cfg.allocator
            target_weights = (plan["hrp_weights"] if allocator_now == "hrp"
                              else plan["kelly_weights"])
            res_comp6 = e_audit.simulate_min_weight_drift_impact(target_weights, {}, cfg.min_weight_drift)
            
            audit_bar.progress(90, text="Comp 7: Validating Regime Architecture...")
            r_audit = RegimeDetectionValidator()
            res_comp7_align = r_audit.validate_regime_classification(regime_history, full_prices)
            res_comp7_persis = r_audit.measure_regime_persistence(regime_history)
            
            audit_bar.progress(100, text="Audit Complete.")
            time.sleep(0.5)
            audit_bar.empty()
            
            # =========================================================
            # RENDER AESTHETIC DASHBOARD
            # =========================================================
            st.divider()
            
            # --- ROW 1: Upstream Integrity ---
            st.markdown("### Upstream Integrity")
            u1, u2 = st.columns(2)
            with u1:
                with st.container(border=True):
                    st.markdown("**1. Feature Integrity & Lookahead**")
                    st.metric("Bias Strength", f"{res_comp1.get('lookahead_bias_strength', 0):.4f}", 
                              delta=res_comp1["interpretation"], delta_color="off")
                    with st.expander("View Sub-metrics"):
                        st.json(res_comp1.get("summary", {}))
            with u2:
                with st.container(border=True):
                    st.markdown("**2. Target Label Construction**")
                    st.metric("Data Quality", res_comp2["interpretation"])
                    dq = res_comp2.get("data_quality", {})
                    st.caption(f"Average Null Ratio: {dq.get('avg_null_ratio', 0):.2%} | Penny Stocks: {dq.get('penny_stocks_detected', 0)}")
            
            # --- ROW 2: Signal Dynamics ---
            st.markdown("### Signal Dynamics")
            s1, s2 = st.columns(2)
            with s1:
                with st.container(border=True):
                    st.markdown("**3. Rank IC vs. Absolute Returns**")
                    ic_trap = res_comp3.get("ic_trap_indicator", {})
                    st.metric("IC Trap Strength", f"{ic_trap.get('trap_strength', 0):.4f}", 
                              delta=ic_trap.get("interpretation", "N/A"), delta_color="off")
                    with st.expander("View Realized Statistics"):
                        st.json(res_comp3.get("realized_statistics", {}))
            with s2:
                with st.container(border=True):
                    st.markdown("**5. Alpha Time Decay**")
                    st.metric("5-Day Decay Rate", f"{res_comp5.get('decay_rate_per_5days', 0):.2%}", 
                              delta=res_comp5.get("interpretation", "N/A"), delta_color="inverse")
                    with st.expander("View IC by Holding Period"):
                        st.json(res_comp5.get("ic_by_holding_period", {}))

            # --- ROW 3: Risk Architecture ---
            st.markdown("### Risk Architecture")
            r1, r2 = st.columns(2)
            with r1:
                with st.container(border=True):
                    st.markdown("**4. Allocator Stress Test (HRP vs Kelly)**")
                    if "error" not in res_comp4:
                        div = res_comp4.get("weight_divergence", {})
                        st.metric("Mean Absolute Divergence", f"{div.get('mean_absolute_diff', 0):.2%}",
                                  help=f"Max divergence on ticker: {div.get('max_divergence_ticker', 'N/A')}")
                        with st.expander("View Concentration Metrics"):
                            st.json(res_comp4.get("concentration_metrics", {}))
                    else:
                        st.error(res_comp4["error"])
            with r2:
                with st.container(border=True):
                    st.markdown("**7. HMM Regime Validation**")
                    if "error" not in res_comp7_align and "error" not in res_comp7_persis:
                        align = res_comp7_align.get("alignment_check", {})
                        st.metric("Market Alignment", align.get("interpretation", "N/A"), 
                                  delta=res_comp7_persis.get("interpretation", "N/A"), delta_color="off")
                        with st.expander("View Regime Statistics"):
                            st.write("**Alignment:**")
                            st.json({k: v for k, v in res_comp7_align.items() if k != "alignment_check"})
                            st.write("**Persistence:**")
                            st.json({k: v for k, v in res_comp7_persis.items() if k not in ["interpretation"]})
                    else:
                        st.error("Regime history insufficient for validation.")

            # --- ROW 4: Execution Layer ---
            st.markdown("### Execution Layer")
            with st.container(border=True):
                st.markdown("**6. Execution Friction Audit**")
                st.metric("Min Weight Drift Setup", res_comp6.get("interpretation", "N/A"),
                          help=f"Evaluated at current cfg.min_weight_drift = {cfg.min_weight_drift:.1%}")
                
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Simulated Skip Rate", f"{res_comp6.get('skip_rate', 0):.1%}")
                ec2.metric("Executed Trades", res_comp6.get('executed_count', 0))
                ec3.metric("Skipped Trades", res_comp6.get('skipped_count', 0))
                st.caption("Simulation assumes a cold start (100% cash) against the current plan's target weights.")

            # --- ROW 5: Export Report ---
            st.divider()
            st.subheader("Export Audit Report")
            st.caption("Hover the box and click 'Copy' in the top right to share with your AI.")
            
            report = "**SYSTEM AUDIT ENGINE REPORT**\n\n"
            
            report += "**1. Upstream Integrity**\n"
            report += f"- **Feature Bias:** {res_comp1.get('interpretation', 'N/A')} (Strength: {res_comp1.get('lookahead_bias_strength', 0):.4f})\n"
            dq = res_comp2.get('data_quality', {})
            report += f"- **Target Label:** {res_comp2.get('interpretation', 'N/A')} (Avg Nulls: {dq.get('avg_null_ratio', 0):.2%}, Penny Stocks: {dq.get('penny_stocks_detected', 0)})\n\n"
            
            report += "**2. Signal Dynamics**\n"
            ic_trap = res_comp3.get('ic_trap_indicator', {})
            report += f"- **IC Trap:** {ic_trap.get('interpretation', 'N/A')} (Strength: {ic_trap.get('trap_strength', 0):.4f})\n"
            report += f"- **Alpha Decay (5d):** {res_comp5.get('interpretation', 'N/A')} (Rate: {res_comp5.get('decay_rate_per_5days', 0):.2%})\n\n"
            
            report += "**3. Risk Architecture**\n"
            if "error" not in res_comp4:
                div = res_comp4.get('weight_divergence', {})
                report += f"- **Allocator Stress:** HRP vs Kelly Divergence {div.get('mean_absolute_diff', 0):.2%}\n"
            else:
                report += f"- **Allocator Stress:** Error - {res_comp4['error']}\n"
            
            if "error" not in res_comp7_align and "error" not in res_comp7_persis:
                align = res_comp7_align.get('alignment_check', {})
                report += f"- **Regime Alignment:** {align.get('interpretation', 'N/A')}\n"
                report += f"- **Regime Persistence:** {res_comp7_persis.get('interpretation', 'N/A')}\n\n"
            else:
                report += f"- **Regime Validation:** Error - Insufficient history\n\n"
                
            report += "**4. Execution Layer**\n"
            report += f"- **Min Weight Drift:** {cfg.min_weight_drift:.1%}\n"
            report += f"- **Friction Impact:** {res_comp6.get('interpretation', 'N/A')} (Skip Rate: {res_comp6.get('skip_rate', 0):.1%})\n"
            
            st.code(report, language="markdown")
