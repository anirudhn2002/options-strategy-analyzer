import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- AUTO-REFRESH ---
# Single knob for how often live data is pulled from Yahoo.
# Every refresh costs ~3 Yahoo requests (history + expiry list + option chain),
# so 30s was ~6 requests/min sustained, which trips Yahoo's throttle.
# Raise this number if you still see rate-limit errors; lower it cautiously.
REFRESH_SECONDS = 120

st_autorefresh(interval=REFRESH_SECONDS * 1000, key="data_refresh")

# --- UI Configuration & Custom CSS ---
st.set_page_config(page_title="Options Strategy Pro", layout="wide")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(30, 30, 30, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Options Strategy Analyzer")
st.markdown("Analyze, visualize, and combine options strategies with real-time market data.")
st.caption(f"Last data refresh: {datetime.now().strftime('%H:%M:%S')}")

ET = ZoneInfo("America/New_York")


# ==========================================================
# --- DATA FETCHING (failures are never cached) ---
# ==========================================================
# Why the old version sometimes said "No options data available":
# when Yahoo throttled a request it quietly returned an EMPTY result instead of
# an error, and st.cache_data stored that empty result for REFRESH_SECONDS.
# Now empty results raise an exception. st.cache_data never caches exceptions,
# so the next run tries again, and the last good data is shown meanwhile.
class NoData(Exception):
    pass


def _retry(fn, tries=3, first_delay=1.5):
    delay, last_err = first_delay, None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # yfinance raises many different error types
            last_err = e
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
    raise last_err


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_stock_data(ticker_symbol):
    def once():
        stock = yf.Ticker(ticker_symbol)  # one Ticker object for both calls
        hist = stock.history(period="5d")
        if hist is None or hist.empty or "Close" not in hist:
            raise NoData("Yahoo returned no price history")
        closes = hist["Close"].dropna()
        if closes.empty:
            raise NoData("Yahoo returned no closing prices")
        expiries = tuple(stock.options or ())
        if not expiries:
            raise NoData("Yahoo returned no option expiries")
        current_spot = float(closes.iloc[-1])
        prev_spot = float(closes.iloc[-2]) if len(closes) >= 2 else current_spot
        return current_spot, prev_spot, expiries
    return _retry(once)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_option_chain(ticker_symbol, expiry_date):
    def once():
        chain = yf.Ticker(ticker_symbol).option_chain(expiry_date)
        calls, puts = chain.calls, chain.puts
        if (calls is None or calls.empty) and (puts is None or puts.empty):
            raise NoData(f"Yahoo returned an empty option chain for {expiry_date}")
        return calls.copy(), puts.copy()
    return _retry(once)


def load_with_fallback(state_key, fn, *args):
    """Returns (data, stale_timestamp_or_None, error_or_None)."""
    try:
        data = fn(*args)
        st.session_state[state_key] = {"data": data, "ts": datetime.now()}
        return data, None, None
    except Exception as e:
        saved = st.session_state.get(state_key)
        if saved is not None:
            return saved["data"], saved["ts"], e
        return None, None, e


def stale_banner(ts, err):
    st.warning(
        f"⚠️ Live Yahoo fetch failed ({type(err).__name__}: {err}). "
        f"Showing the last good data from **{ts.strftime('%H:%M:%S')}**. "
        f"It will retry automatically on the next refresh."
    )


# ==========================================================
# --- QUOTE CLEANUP (after-hours bid/ask of 0 or NaN) ---
# ==========================================================
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) and v > 0 else 0.0


def quote(row):
    """Clean prices for one option row. Falls back to the last trade when bid/ask are missing."""
    bid, ask, last = _num(row.get("bid")), _num(row.get("ask")), _num(row.get("lastPrice"))
    live = bid > 0 and ask > 0 and ask >= bid
    mid = (bid + ask) / 2 if live else (last or bid or ask)
    return {
        "bid": bid, "ask": ask, "last": last, "mid": mid, "live": live,
        "sell": bid if bid > 0 else mid,   # what you receive when shorting
        "buy": ask if ask > 0 else mid,    # what you pay when buying
        "sell_live": bid > 0, "buy_live": ask > 0,
    }


# ==========================================================
# --- BLACK SCHOLES CORE MATH ---
# ==========================================================
def bs_price(S, K, T, r, q, sigma, opt):
    if sigma is None or sigma <= 0.0 or T <= 0.0: return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)


def implied_vol(price, S, K, T, r, q, opt):
    if price is None or not np.isfinite(price) or price <= 0:
        return None
    lo, hi = 1e-4, 10.0
    try:
        f_lo = bs_price(S, K, T, r, q, lo, opt) - price
        f_hi = bs_price(S, K, T, r, q, hi, opt) - price
        if f_lo * f_hi > 0:  # price below intrinsic or absurdly high: no solution
            return None
        return brentq(lambda s: bs_price(S, K, T, r, q, s, opt) - price, lo, hi)
    except Exception:
        return None


def iv_with_fallback(price, row, S, K, T, r, q, opt):
    """IV from the given price; if that fails (stale after-hours quotes), use Yahoo's IV column.
    Returns (iv, source) with source in {'quote', 'yahoo', None}."""
    iv = implied_vol(price, S, K, T, r, q, opt)
    if iv is not None:
        return iv, "quote"
    y = _num(row.get("impliedVolatility"))
    if 0.01 < y < 10:
        return y, "yahoo"
    return None, None


def calculate_delta(S, K, T, r, q, sigma, opt):
    if sigma is None or sigma <= 0.0 or T <= 0.0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    if opt == "call":
        return np.exp(-q*T) * norm.cdf(d1)
    else:
        return np.exp(-q*T) * (norm.cdf(d1) - 1.0)


def time_to_expiry(expiry):
    """Years until 4:00pm New York time on expiry day (floored at 1 hour)."""
    exp_close = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=16, tzinfo=ET)
    secs = (exp_close - datetime.now(timezone.utc)).total_seconds()
    return max(secs / (365 * 24 * 3600), 1 / (365 * 24))


# ==========================================================
# --- EXPIRY PAYOFF ENGINE ---
# A position is a list of components:
#   {"kind": "call"|"put"|"stock", "k": strike, "sign": +1 long / -1 short, "qty": q, "entry": price}
# Everything is per-option (per-share) basis, no x100 multiplier.
# ==========================================================
def payoff_of(comps, s):
    s = np.asarray(s, dtype=float)
    total = np.zeros_like(s)
    for c in comps:
        if c["kind"] == "call":
            v = np.maximum(s - c["k"], 0.0)
        elif c["kind"] == "put":
            v = np.maximum(c["k"] - s, 0.0)
        else:
            v = s
        total += c["sign"] * c["qty"] * (v - c["entry"])
    return total


def stock_comp(shares, price):
    return {"kind": "stock", "k": None, "sign": 1, "qty": float(shares), "entry": float(price)}


def payoff_stats(comps):
    """Exact breakevens / max profit / max loss (expiry payoff is piecewise linear, kinks at strikes)."""
    ks = sorted({float(c["k"]) for c in comps if c["kind"] != "stock" and c["k"] > 0})
    xs = np.array([0.0] + ks if ks else [0.0, 1.0])
    ys = payoff_of(comps, xs)
    slope_right = sum(c["sign"] * c["qty"] for c in comps if c["kind"] in ("call", "stock"))
    tol = 1e-9
    bes = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if abs(y0) < tol and xs[i] > 0:
            bes.append(xs[i])
        elif y0 * y1 < 0:
            bes.append(xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0))
    if abs(ys[-1]) < tol and xs[-1] > 0:
        bes.append(xs[-1])
    elif abs(slope_right) > tol and ys[-1] * slope_right < 0:
        bes.append(xs[-1] - ys[-1] / slope_right)
    bes = sorted({round(b, 2) for b in bes})
    max_p = float("inf") if slope_right > tol else float(ys.max())
    max_l = float("-inf") if slope_right < -tol else float(ys.min())
    return bes, max_p, max_l


def fmt_money(v):
    if v == float("inf"): return "Unlimited ↑"
    if v == float("-inf"): return "Unlimited ↓"
    return f"${v:,.2f}"


def stats_table(series, spot):
    rows = []
    for s in series:
        bes, mp, ml = payoff_stats(s["comps"])
        rows.append({
            "Line": s["name"],
            "Breakevens": ", ".join(f"${b:,.2f}" for b in bes) if bes else "None",
            "Max Profit": fmt_money(mp),
            "Max Loss": fmt_money(ml),
            f"P&L if expires at spot (${spot:,.2f})": f"${float(payoff_of(s['comps'], [spot])[0]):,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ==========================================================
# --- PAYOFF CHART ---
# theme=None is the fix for the "lines look identical / unhedged plot is empty" bug:
# Streamlit was replacing the plotly_dark template with its own light theme, so the
# white "Unhedged" line was drawn white-on-white and was invisible.
# ==========================================================
def _hex_rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def _add_fill(fig, x, y, name, yaxis="y"):
    fig.add_trace(go.Scatter(x=x, y=np.where(y >= 0, y, 0), fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.18)',
                             line=dict(width=0), hoverinfo='skip', showlegend=False, legendgroup=name, yaxis=yaxis))
    fig.add_trace(go.Scatter(x=x, y=np.where(y < 0, y, 0), fill='tozeroy', fillcolor='rgba(255, 60, 60, 0.18)',
                             line=dict(width=0), hoverinfo='skip', showlegend=False, legendgroup=name, yaxis=yaxis))


def _add_line(fig, x, s, yaxis="y", show_legend=True):
    fig.add_trace(go.Scatter(
        x=x, y=s["y"], name=s["name"], legendgroup=s["name"], yaxis=yaxis, showlegend=show_legend,
        line=dict(color=s["color"], width=s.get("width", 3), dash=s.get("dash", "solid")),
        opacity=s.get("opacity", 1.0),
        hovertemplate="%{y:$,.2f}",
    ))
    bes, _, _ = payoff_stats(s["comps"])
    bes = [b for b in bes if x[0] <= b <= x[-1]]
    if bes:
        fig.add_trace(go.Scatter(
            x=bes, y=[0] * len(bes), mode="markers", legendgroup=s["name"], showlegend=False, yaxis=yaxis,
            marker=dict(color=s["color"], size=10, symbol="diamond", line=dict(color="black", width=1)),
            name=f"BE · {s['name']}", hovertemplate="breakeven %{x:$,.2f}",
        ))


def render_payoff_chart(x, series, layout_mode, spot, strikes, chart_key, height=600):
    if not series:
        st.info("Pick at least one line to plot.")
        return
    fig = go.Figure()
    stacked = layout_mode == "Stacked panels" and len(series) > 1

    if stacked:
        n, gap = len(series), 0.05
        h = (1 - gap * (n - 1)) / n
        for i, s in enumerate(series):
            ax = "" if i == 0 else str(i + 1)
            top = 1 - i * (h + gap)
            fig.update_layout(**{f"yaxis{ax}": dict(domain=[top - h, top], tickprefix="$", zeroline=True,
                                                    zerolinecolor="rgba(200,200,200,0.6)", zerolinewidth=1.5)})
            _add_fill(fig, x, s["y"], s["name"], yaxis="y" + ax)
            _add_line(fig, x, s, yaxis="y" + ax)
            fig.add_annotation(text=f"<b>{s['name']}</b>", xref="paper", yref="paper", x=0.005, y=top,
                               xanchor="left", yanchor="top", showarrow=False, font=dict(color=s["color"], size=13))
        fig.update_layout(height=max(260 * n + 60, 450), hoversubplots="axis", showlegend=False,
                          xaxis=dict(anchor=f"y{n}"))
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        if len(series) == 1:  # shading only when it's unambiguous whose profit/loss it is
            _add_fill(fig, x, series[0]["y"], series[0]["name"])
        for s in series:
            _add_line(fig, x, s)
        fig.update_layout(height=height, yaxis=dict(tickprefix="$"),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.4)"))

    for k in strikes:
        fig.add_shape(type="line", x0=k, x1=k, xref="x", y0=0, y1=1, yref="paper",
                      line=dict(dash="dot", color="rgba(255, 255, 255, 0.35)"))
        fig.add_annotation(x=k, y=1.0, xref="x", yref="paper", text=f"K={k:g}", showarrow=False,
                           yanchor="bottom", font=dict(size=10, color="rgba(255,255,255,0.7)"))
    fig.add_shape(type="line", x0=spot, x1=spot, xref="x", y0=0, y1=1, yref="paper",
                  line=dict(color="#FFD54F", width=1.5))
    fig.add_annotation(x=spot, y=0, xref="x", yref="paper", text=f"Spot {spot:,.2f}", showarrow=False,
                       yanchor="bottom", xanchor="left", font=dict(size=10, color="#FFD54F"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=CHART_FONT, hoverlabel=CHART_HOVER,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="Underlying Price at Expiry", hoverformat="$,.2f"),
        uirevision=chart_key,  # keeps zoom and legend clicks through auto-refreshes
    )
    st.plotly_chart(fig, theme=None, key=chart_key, width="stretch")


# set explicitly so the dark look never depends on the Streamlit theme
CHART_BG = "#0E1117"
CHART_FONT = dict(color="#E6E6E6")
CHART_HOVER = dict(bgcolor="#1E222B", font=dict(color="#FFFFFF"))

LINE_STYLES = {
    # drawn in this order; the wide translucent line sits underneath so overlapping lines stay visible
    "Hedged (Real Price)": dict(color="#00E676", width=7, opacity=0.45),
    "Unhedged Options": dict(color="#FFB300", width=3),
    "Hedged (Mid Price)": dict(color="#29B6F6", width=2.5, dash="dash"),
    "Custom Shares": dict(color="#FF4FD8", width=3, dash="dashdot"),
}
# ==========================================================


# --- Ticker Selection ---
with st.container():
    col_t1, col_t2 = st.columns([1, 2])
    popular_tickers = ["AAPL", "TSLA", "SPY", "QQQ", "NVDA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "Other..."]

    selected_ticker = col_t1.selectbox("Select Asset Ticker", popular_tickers, key="app_ticker")

    if selected_ticker == "Other...":
        ticker = col_t2.text_input("Enter Custom Ticker", value="NFLX", key="custom_ticker").upper().strip()
    else:
        ticker = selected_ticker

if "active_ticker" not in st.session_state:
    st.session_state["active_ticker"] = ticker
elif st.session_state["active_ticker"] != ticker:
    keys_to_clear = [k for k in st.session_state.keys() if "strike" in k or "_k" in k]
    for k in keys_to_clear:
        del st.session_state[k]
    st.session_state["active_ticker"] = ticker


if ticker:
    stock_data, stale_ts, fetch_err = load_with_fallback(f"lastgood_stock_{ticker}", fetch_stock_data, ticker)

    if stock_data is None:
        st.error(
            f"No options data available for **{ticker}** right now ({type(fetch_err).__name__}: {fetch_err}). "
            "This is usually Yahoo throttling. Nothing bad was cached, so the next attempt starts fresh. "
            "If the ticker is right, wait a few seconds and retry."
        )
        st.button("Retry now", key="retry_fetch")  # clicking reruns the script, which re-fetches
        st.stop()
    if stale_ts is not None:
        stale_banner(stale_ts, fetch_err)

    current_spot, prev_spot, expiries = stock_data
    spot_change = current_spot - prev_spot
    spot_change_pct = (spot_change / prev_spot) * 100 if prev_spot else 0.0

    st.divider()
    col_spot, col_exp, col_r, col_q = st.columns([1, 2, 0.7, 0.7])
    with col_spot:
        st.metric(
            label=f"Current Spot Price: {ticker}",
            value=f"${current_spot:.2f}",
            delta=f"{spot_change:.2f} ({spot_change_pct:.2f}%)"
        )

    with col_exp:
        if "app_expiry" in st.session_state and st.session_state["app_expiry"] not in expiries:
            del st.session_state["app_expiry"]

        expiry = st.selectbox("Select Expiry Date", expiries, key="app_expiry")

    r_pct = col_r.number_input("Risk-free rate (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.25,
                               key="risk_free_pct", help="Used for IV and delta. Default 8%.")
    q_pct = col_q.number_input("Dividend yield (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.25,
                               key="div_yield_pct", help="Continuous dividend yield used for IV and delta.")

    chain_data, chain_stale_ts, chain_err = load_with_fallback(
        f"lastgood_chain_{ticker}_{expiry}", fetch_option_chain, ticker, expiry)
    if chain_data is None:
        st.error(f"Could not load the option chain for {ticker} {expiry} ({type(chain_err).__name__}: {chain_err}). "
                 "Usually Yahoo throttling. Retry in a few seconds.")
        st.button("Retry now", key="retry_chain")
        st.stop()
    if chain_stale_ts is not None:
        stale_banner(chain_stale_ts, chain_err)

    calls, puts = chain_data
    S = current_spot
    T = time_to_expiry(expiry)
    r, q = r_pct / 100.0, q_pct / 100.0
    col_exp.caption(f"Time to expiry: {T * 365:.2f} days (to 4:00pm ET close)")

    # ==================================================
    # --- 1. Short Straddle Analysis ---
    # ==================================================
    st.markdown("### Short Straddle Analysis")
    common_strikes = sorted(list(set(calls['strike']).intersection(set(puts['strike']))))

    if common_strikes:
        closest_strike = min(common_strikes, key=lambda x: abs(x - S))

        if "straddle_strike" in st.session_state and st.session_state["straddle_strike"] not in common_strikes:
            del st.session_state["straddle_strike"]
        if "straddle_strike" not in st.session_state:
            st.session_state["straddle_strike"] = closest_strike

        col_st_input, col_st_metric = st.columns(2)
        selected_strike = col_st_input.selectbox("Select Strike Price for Straddle", common_strikes, key="straddle_strike")

        cq = quote(calls[calls['strike'] == selected_strike].iloc[0])
        pq = quote(puts[puts['strike'] == selected_strike].iloc[0])

        premium_collected = cq["sell"] + pq["sell"]
        col_st_metric.metric("Total Premium Collected (Bid)", f"${premium_collected:.2f}")
        if not (cq["sell_live"] and pq["sell_live"]):
            col_st_metric.caption("⚠️ Bid missing (market closed?). Using last traded price for the missing side.")

        be_low, be_high = selected_strike - premium_collected, selected_strike + premium_collected
        s_min_strad, s_max_strad = be_low * 0.95, be_high * 1.05
        s_vals = np.linspace(s_min_strad, s_max_strad, 400)
        payoff = premium_collected - np.abs(s_vals - selected_strike)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=[s_min_strad, s_max_strad], y=[0, 0], name='Zero Line', line=dict(color='gray', dash='dash'), hoverinfo='skip'))
        fig1.add_trace(go.Scatter(x=s_vals, y=payoff, name='Payoff', line=dict(color='white', width=3), hovertemplate="%{y:$,.2f}"))
        fig1.add_trace(go.Scatter(x=s_vals, y=np.where(payoff >= 0, payoff, 0), fill='tozeroy', fillcolor='rgba(0, 255, 100, 0.25)', line=dict(width=0), name="Profit Zone", hoverinfo='skip'))
        fig1.add_trace(go.Scatter(x=s_vals, y=np.where(payoff < 0, payoff, 0), fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.25)', line=dict(width=0), name="Loss Zone", hoverinfo='skip'))

        fig1.add_annotation(x=be_low, y=0, text=f"Breakeven: ${be_low:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white", ax=-50, ay=-40)
        fig1.add_annotation(x=be_high, y=0, text=f"Breakeven: ${be_high:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white", ax=50, ay=-40)

        fig1.update_layout(title="Short Straddle Payoff Curve", xaxis_title="Underlying Price at Expiry", yaxis_title="Profit & Loss ($)",
                           height=450, template="plotly_dark", paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=CHART_FONT, hoverlabel=CHART_HOVER, hovermode="x unified", xaxis=dict(hoverformat="$,.2f"),
                           uirevision="straddle")
        st.plotly_chart(fig1, theme=None, key="straddle_chart", width="stretch")

    # ==================================================
    # --- 2. Single Option Data Inspector ---
    # ==================================================
    st.divider()
    st.markdown("### Option Data Inspector")
    col_call, col_put = st.columns(2)

    call_strikes = sorted(calls['strike'].tolist())
    put_strikes = sorted(puts['strike'].tolist())

    def inspector(col, label, strikes, df, key, opt):
        with col:
            st.markdown(f"#### {label} Option")
            if not strikes:
                st.info(f"No {label.lower()}s listed for this expiry.")
                return
            closest = min(strikes, key=lambda x: abs(x - S))
            if key in st.session_state and st.session_state[key] not in strikes:
                del st.session_state[key]
            if key not in st.session_state:
                st.session_state[key] = closest

            k_sel = st.selectbox(f"Select {label} Strike", strikes, key=key)
            row = df[df['strike'] == k_sel].iloc[0]
            qt = quote(row)
            iv, src = iv_with_fallback(qt["mid"], row, S, k_sel, T, r, q, opt)

            m1, m2, m3 = st.columns(3)
            m1.metric("Bid", f"${qt['bid']:.2f}")
            m2.metric("Ask", f"${qt['ask']:.2f}")
            m3.metric("Market Implied Vol (Mid)", f"{iv*100:.2f}%" if iv else "N/A")
            if not qt["live"]:
                st.caption(f"⚠️ No live bid/ask. Mid uses last trade (\\${qt['last']:.2f}).")
            if src == "yahoo":
                st.caption("⚠️ Couldn't solve IV from the price. Showing Yahoo's IV instead.")

    inspector(col_call, "Call", call_strikes, calls, "insp_call_k", "call")
    inspector(col_put, "Put", put_strikes, puts, "insp_put_k", "put")

    # ==================================================
    # --- 3. Custom Strategy Builder (Dynamic Legs) ---
    # ==================================================
    st.divider()
    st.markdown("### Combination Strategy Builder")

    num_legs = st.number_input("How many strategy legs?", min_value=1, value=4, step=1)

    all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
    closest_strike_all = min(all_strikes, key=lambda x: abs(x - S)) if all_strikes else 0

    legs = []

    for row_idx in range(0, num_legs, 4):
        cols = st.columns(4)
        for col_idx in range(4):
            i = row_idx + col_idx
            if i < num_legs:
                with cols[col_idx]:
                    st.markdown(f"##### Leg {i+1}")
                    if f"leg_{i}_k" in st.session_state and st.session_state[f"leg_{i}_k"] not in all_strikes:
                        del st.session_state[f"leg_{i}_k"]

                    if f"leg_{i}_t" not in st.session_state: st.session_state[f"leg_{i}_t"] = "Call"
                    if f"leg_{i}_p" not in st.session_state: st.session_state[f"leg_{i}_p"] = "Short"
                    if f"leg_{i}_k" not in st.session_state: st.session_state[f"leg_{i}_k"] = closest_strike_all
                    if f"leg_{i}_qty" not in st.session_state: st.session_state[f"leg_{i}_qty"] = 1 if i < 2 else 0

                    t = st.selectbox("Type", ["Call", "Put"], key=f"leg_{i}_t")
                    pos = st.selectbox("Action", ["Short", "Long"], key=f"leg_{i}_p")
                    k = st.selectbox("Strike", all_strikes, key=f"leg_{i}_k")
                    qty = st.number_input("Quantity", min_value=0, step=1, key=f"leg_{i}_qty")

                    legs.append({"type": t, "pos": pos, "k": k, "qty": qty})

    net_cf = 0.0
    net_delta_mid = 0.0
    net_delta_real = 0.0
    active_strikes = []
    new_comps = []            # new legs at today's real (bid/ask) prices
    stale_legs, iv_issues = [], []

    for n_leg, leg in enumerate(legs, start=1):
        if leg["qty"] == 0: continue
        df = calls if leg["type"] == "Call" else puts
        row = df[df['strike'] == leg["k"]]
        if row.empty: continue
        row = row.iloc[0]
        opt = leg["type"].lower()

        active_strikes.append(leg["k"])
        qt = quote(row)
        real_price = qt["sell"] if leg["pos"] == "Short" else qt["buy"]
        if not (qt["sell_live"] if leg["pos"] == "Short" else qt["buy_live"]):
            stale_legs.append(n_leg)

        iv_mid, src_mid = iv_with_fallback(qt["mid"], row, S, leg["k"], T, r, q, opt)
        iv_real, src_real = iv_with_fallback(real_price, row, S, leg["k"], T, r, q, opt)
        if iv_real is None or src_real == "yahoo" or src_mid == "yahoo" or iv_mid is None:
            iv_issues.append(n_leg)
        if iv_real is None:
            iv_real = iv_mid  # better than silently using delta = 0

        leg_delta_mid = calculate_delta(S, leg["k"], T, r, q, iv_mid, opt)
        leg_delta_real = calculate_delta(S, leg["k"], T, r, q, iv_real, opt)

        sign = 1 if leg["pos"] == "Long" else -1

        net_cf += (real_price * -sign) * leg["qty"]
        net_delta_mid += (leg_delta_mid * sign) * leg["qty"]
        net_delta_real += (leg_delta_real * sign) * leg["qty"]
        new_comps.append({"kind": opt, "k": float(leg["k"]), "sign": sign, "qty": float(leg["qty"]), "entry": real_price})

    hedge_shares_mid = -net_delta_mid
    hedge_shares_real = -net_delta_real

    crit_points = active_strikes + [S]
    s_min_c = min(crit_points) * 0.75
    s_max_c = max(crit_points) * 1.25
    test_s = np.linspace(s_min_c, s_max_c, 5000)

    if new_comps:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("##### Hedging & Delta Summary")
        st.caption("Compare the theoretical assumptions of Mid-Price hedging against Real-Price (Bid/Ask) hedging.")

        c_sum1, c_sum2, c_sum3, c_sum4, c_sum5 = st.columns(5)
        c_sum1.metric("Net Premium Collected", f"${net_cf:.2f}")
        c_sum2.metric("Net Delta (Mid)", f"{net_delta_mid:.4f}")
        c_sum3.metric("Net Delta (Real)", f"{net_delta_real:.4f}")
        c_sum4.metric("Hedge Shares (Mid)", f"{hedge_shares_mid:.4f}")
        c_sum5.metric("Hedge Shares (Real)", f"{hedge_shares_real:.4f}")
        if stale_legs:
            st.caption(f"⚠️ Leg(s) {', '.join(map(str, stale_legs))}: no live bid/ask (market closed?). Using last traded price.")
        if iv_issues:
            st.caption(f"⚠️ Leg(s) {', '.join(map(str, iv_issues))}: IV couldn't be solved from the quote. "
                       "Used mid IV or Yahoo's IV instead, so delta may be less accurate.")

        c_cs1, c_cs2, c_cs3 = st.columns([1.2, 1, 2.8])
        custom_shares = c_cs1.number_input(
            "Custom underlying shares", value=0.0, step=0.05, format="%.4f", key="custom_shares",
            help="Per-option basis, same units as Hedge Shares. Positive = long stock, negative = short. "
                 "Bought or sold at today's spot. Add 'Custom Shares' to the lines below to plot it.")
        c_cs2.metric("Net Delta w/ Custom Shares", f"{net_delta_real + custom_shares:+.4f}")
        c_cs3.caption("Lean the book long or short delta on purpose. Compare it with the delta-neutral hedge "
                      "by showing both lines.")
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("#### Combined Strategy Payoff")
        c_sel, c_lay = st.columns([3, 1])
        builder_lines = c_sel.multiselect(
            "Lines to show", list(LINE_STYLES.keys()),
            default=["Hedged (Real Price)", "Unhedged Options", "Hedged (Mid Price)"], key="builder_lines")
        builder_layout = c_lay.radio("Layout", ["Single chart", "Stacked panels"], key="builder_layout", horizontal=True)
        st.caption("💡 **Tip:** Click legend labels to hide or show lines. Diamonds mark breakevens. "
                   "Profit/loss shading appears when a single line is shown, and always in stacked panels.")

        comps_by_line = {
            "Unhedged Options": new_comps,
            "Hedged (Mid Price)": new_comps + [stock_comp(hedge_shares_mid, S)],
            "Hedged (Real Price)": new_comps + [stock_comp(hedge_shares_real, S)],
            "Custom Shares": new_comps + [stock_comp(custom_shares, S)],
        }
        series = []
        for name in LINE_STYLES:  # fixed draw order
            if name in builder_lines:
                series.append({"name": name, "comps": comps_by_line[name],
                               "y": payoff_of(comps_by_line[name], test_s), **LINE_STYLES[name]})

        render_payoff_chart(test_s, series, builder_layout, S, sorted(set(active_strikes)), "builder_chart")
        if series:
            st.markdown("##### Breakevens & Risk at Expiry")
            stats_table(series, S)
    else:
        custom_shares = 0.0
        st.info("Set at least one Leg's Quantity > 0 to see the Strategy Plots.")

    # ==================================================
    # --- 4. My Existing Positions (entered at my fill prices) ---
    # ==================================================
    st.divider()
    st.markdown("### My Existing Positions")
    st.caption("Enter positions you already hold at the price you actually traded them. The expiry payoff uses "
               "your fill prices, not today's quotes. Same expiry as selected above, per-option basis. "
               "The combined lines add the new legs from the builder above (priced at today's bid/ask).")

    n_held = st.number_input("How many existing positions?", min_value=0, value=1, step=1, key="held_n")

    held_comps, held_missing_px = [], []
    held_delta, held_open_pnl = 0.0, 0.0
    held_iv_issue = []

    for row_idx in range(0, int(n_held), 4):
        cols = st.columns(4)
        for col_idx in range(4):
            i = row_idx + col_idx
            if i >= n_held:
                continue
            with cols[col_idx]:
                st.markdown(f"##### Position {i+1}")
                if f"held_{i}_strike" in st.session_state and st.session_state[f"held_{i}_strike"] not in all_strikes:
                    del st.session_state[f"held_{i}_strike"]
                if f"held_{i}_type" not in st.session_state: st.session_state[f"held_{i}_type"] = "Call"
                if f"held_{i}_side" not in st.session_state: st.session_state[f"held_{i}_side"] = "Short"
                if f"held_{i}_strike" not in st.session_state: st.session_state[f"held_{i}_strike"] = closest_strike_all
                if f"held_{i}_qty" not in st.session_state: st.session_state[f"held_{i}_qty"] = 0.0

                h_type = st.selectbox("Type", ["Call", "Put", "Stock"], key=f"held_{i}_type")
                h_side = st.selectbox("Action", ["Short", "Long"], key=f"held_{i}_side")
                if h_type != "Stock":
                    h_k = st.selectbox("Strike", all_strikes, key=f"held_{i}_strike")
                else:
                    h_k = None
                h_px = st.number_input("Your entry price (per share)", min_value=0.0, value=None, step=0.01,
                                       format="%.2f", key=f"held_{i}_px", placeholder="your fill price")
                h_qty = st.number_input("Quantity" if h_type != "Stock" else "Shares (per-option basis)",
                                        min_value=0.0, step=1.0, format="%.2f", key=f"held_{i}_qty")

                if h_qty <= 0:
                    continue
                if h_px is None:
                    held_missing_px.append(i + 1)
                    continue
                sign = 1 if h_side == "Long" else -1
                if h_type == "Stock":
                    held_comps.append({"kind": "stock", "k": None, "sign": sign, "qty": float(h_qty), "entry": float(h_px)})
                    held_delta += sign * h_qty
                    held_open_pnl += sign * h_qty * (S - h_px)
                    st.caption(f"Open P&L now: \\${sign * h_qty * (S - h_px):,.2f}")
                else:
                    opt = h_type.lower()
                    df = calls if h_type == "Call" else puts
                    row = df[df['strike'] == h_k]
                    held_comps.append({"kind": opt, "k": float(h_k), "sign": sign, "qty": float(h_qty), "entry": float(h_px)})
                    if row.empty:
                        held_iv_issue.append(i + 1)
                        continue
                    row = row.iloc[0]
                    qt = quote(row)
                    iv, _src = iv_with_fallback(qt["mid"], row, S, h_k, T, r, q, opt)
                    if iv is None:
                        held_iv_issue.append(i + 1)
                    held_delta += sign * h_qty * calculate_delta(S, h_k, T, r, q, iv, opt)
                    pos_pnl = sign * h_qty * (qt["mid"] - h_px)
                    held_open_pnl += pos_pnl
                    st.caption(f"Mark (mid): \\${qt['mid']:.2f} · Open P&L: \\${pos_pnl:,.2f}")

    if held_missing_px:
        st.warning(f"Enter your entry price for position(s) {', '.join(map(str, held_missing_px))} to include them.")

    if held_comps:
        book_delta = held_delta + net_delta_real
        book_hedge = -book_delta

        st.markdown("<hr>", unsafe_allow_html=True)
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Open P&L (at current mid)", f"${held_open_pnl:,.2f}")
        h2.metric("Net Delta (Existing)", f"{held_delta:.4f}")
        h3.metric("Net Delta (Existing + New)", f"{book_delta:.4f}")
        h4.metric("Hedge Shares (Existing + New)", f"{book_hedge:.4f}")
        book_custom = h5.number_input("Custom shares for combined book", value=0.0, step=0.05, format="%.4f",
                                      key="book_custom_shares",
                                      help="Per-option basis, bought or sold at today's spot. Positive = long.")
        if held_iv_issue:
            st.caption(f"⚠️ Position(s) {', '.join(map(str, held_iv_issue))}: no usable IV, so delta for those is treated as 0.")

        plus_new = " + New Legs" if new_comps else ""
        book_lines = {
            "Existing Positions": (held_comps, dict(color="#FFB300", width=3)),
            f"Existing{plus_new} + Delta Hedge": (held_comps + new_comps + [stock_comp(book_hedge, S)],
                                                  dict(color="#00E676", width=7, opacity=0.45)),
            f"Existing{plus_new} + Custom Shares": (held_comps + new_comps + [stock_comp(book_custom, S)],
                                                    dict(color="#FF4FD8", width=3, dash="dashdot")),
        }
        if new_comps:
            book_lines = {
                "Existing Positions": book_lines["Existing Positions"],
                "New Legs Only": (new_comps, dict(color="#90A4AE", width=2, dash="dot")),
                "Existing + New Legs": (held_comps + new_comps, dict(color="#29B6F6", width=3)),
                **{k: v for k, v in book_lines.items() if k != "Existing Positions"},
            }
        # draw wide translucent hedge line first so the others stay visible on top
        order = sorted(book_lines, key=lambda n: 0 if "Delta Hedge" in n else 1)

        st.markdown("#### Portfolio Payoff at Expiry")
        c_sel2, c_lay2 = st.columns([3, 1])
        default_book = [n for n in book_lines if n in ("Existing Positions", "Existing + New Legs")]
        # the available line names change when the builder legs change, so the widget key tracks that
        book_sel = c_sel2.multiselect("Lines to show", list(book_lines.keys()), default=default_book,
                                      key=f"book_lines_{'new' if new_comps else 'solo'}")
        book_layout = c_lay2.radio("Layout", ["Single chart", "Stacked panels"], key="book_layout", horizontal=True)

        book_strikes = sorted({c["k"] for c in held_comps + new_comps if c["kind"] != "stock"})
        pts = book_strikes + [S] + [c["entry"] for c in held_comps if c["kind"] == "stock"]
        book_x = np.linspace(min(pts) * 0.75, max(pts) * 1.25, 5000)

        book_series = [{"name": n, "comps": book_lines[n][0], "y": payoff_of(book_lines[n][0], book_x), **book_lines[n][1]}
                       for n in order if n in book_sel]
        render_payoff_chart(book_x, book_series, book_layout, S, book_strikes, "book_chart")
        if book_series:
            st.markdown("##### Breakevens & Risk at Expiry")
            stats_table(book_series, S)
    elif n_held > 0 and not held_missing_px:
        st.info("Set a position's Quantity > 0 and enter your entry price to plot your portfolio.")
