import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np
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


# ==========================================================
# --- CACHED DATA FETCHING ---
# ==========================================================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="5d")
    expiries = stock.options

    if len(hist) >= 2:
        current_spot = float(hist["Close"].iloc[-1])
        prev_spot = float(hist["Close"].iloc[-2])
    elif len(hist) == 1:
        current_spot = float(hist["Close"].iloc[-1])
        prev_spot = current_spot
    else:
        return None, None, ()

    return current_spot, prev_spot, expiries

@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_option_chain(ticker_symbol, expiry_date):
    stock = yf.Ticker(ticker_symbol)
    chain = stock.option_chain(expiry_date)
    return chain.calls, chain.puts


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
    try:
        if price <= 0: return None
        return brentq(lambda s: bs_price(S,K,T,r,q,s,opt)-price, 1e-6, 10.0)
    except:
        return None

def calculate_delta(S, K, T, r, q, sigma, opt):
    if sigma is None or sigma <= 0.0 or T <= 0.0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    if opt == "call":
        return np.exp(-q*T) * norm.cdf(d1)
    else:
        return np.exp(-q*T) * (norm.cdf(d1) - 1.0)
# ==========================================================


# --- Ticker Selection ---
with st.container():
    col_t1, col_t2 = st.columns([1, 2])
    popular_tickers = ["AAPL", "TSLA", "SPY", "QQQ", "NVDA", "AMD", "AMZN", "MSFT", "META", "GOOGL", "Other..."]

    selected_ticker = col_t1.selectbox("Select Asset Ticker", popular_tickers, key="app_ticker")

    if selected_ticker == "Other...":
        ticker = col_t2.text_input("Enter Custom Ticker", value="NFLX", key="custom_ticker").upper()
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
    current_spot, prev_spot, expiries = fetch_stock_data(ticker)

    if current_spot is None:
        st.error("Could not fetch spot price. Please check the ticker.")
        st.stop()

    if expiries:
        spot_change = current_spot - prev_spot
        spot_change_pct = (spot_change / prev_spot) * 100 if prev_spot else 0.0

        st.divider()
        col_spot, col_exp = st.columns([1, 2])
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

        calls, puts = fetch_option_chain(ticker, expiry)
        S = current_spot
        T = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.utcnow()).days / 365, 1/365)
        r, q = 0.06, 0.0

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

            c_row = calls[calls['strike'] == selected_strike].iloc[0]
            p_row = puts[puts['strike'] == selected_strike].iloc[0]

            premium_collected = c_row['bid'] + p_row['bid']
            col_st_metric.metric("Total Premium Collected (Bid)", f"${premium_collected:.2f}")

            be_low, be_high = selected_strike - premium_collected, selected_strike + premium_collected
            s_min_strad, s_max_strad = be_low * 0.95, be_high * 1.05
            s_vals = np.linspace(s_min_strad, s_max_strad, 400)
            payoff = premium_collected - np.abs(s_vals - selected_strike)

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=[s_min_strad, s_max_strad], y=[0, 0], name='Zero Line', line=dict(color='gray', dash='dash')))
            fig1.add_trace(go.Scatter(x=s_vals, y=payoff, name='Payoff', line=dict(color='white', width=3)))
            fig1.add_trace(go.Scatter(x=s_vals, y=np.where(payoff >= 0, payoff, 0), fill='tozeroy', fillcolor='rgba(0, 255, 100, 0.25)', line=dict(width=0), name="Profit Zone"))
            fig1.add_trace(go.Scatter(x=s_vals, y=np.where(payoff < 0, payoff, 0), fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.25)', line=dict(width=0), name="Loss Zone"))

            fig1.add_annotation(x=be_low, y=0, text=f"Breakeven: ${be_low:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white", ax=-50, ay=-40)
            fig1.add_annotation(x=be_high, y=0, text=f"Breakeven: ${be_high:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="white", ax=50, ay=-40)

            fig1.update_layout(title="Short Straddle Payoff Curve", xaxis_title="Underlying Price at Expiry", yaxis_title="Profit & Loss ($)", height=450, template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)

        # ==================================================
        # --- 2. Single Option Data Inspector ---
        # ==================================================
        st.divider()
        st.markdown("### Option Data Inspector")
        col_call, col_put = st.columns(2)

        call_strikes = sorted(calls['strike'].tolist())
        put_strikes = sorted(puts['strike'].tolist())

        with col_call:
            st.markdown("#### Call Option")
            c_closest = min(call_strikes, key=lambda x: abs(x - S)) if call_strikes else 0
            if "insp_call_k" in st.session_state and st.session_state["insp_call_k"] not in call_strikes:
                del st.session_state["insp_call_k"]
            if "insp_call_k" not in st.session_state:
                st.session_state["insp_call_k"] = c_closest

            c_strike = st.selectbox("Select Call Strike", call_strikes, key="insp_call_k")
            c_row = calls[calls['strike'] == c_strike].iloc[0]

            c_bid, c_ask = c_row.get("bid", 0.0), c_row.get("ask", 0.0)
            c_mid = (c_bid + c_ask) / 2 if (c_bid > 0 and c_ask > 0) else c_row['lastPrice']
            c_iv = implied_vol(c_mid, S, c_strike, T, r, q, 'call')

            c1, c2, c4 = st.columns(3)
            c1.metric("Bid", f"${c_bid:.2f}")
            c2.metric("Ask", f"${c_ask:.2f}")
            c4.metric("Market Implied Vol (Mid)", f"{c_iv*100:.2f}%" if c_iv else "N/A")

        with col_put:
            st.markdown("#### Put Option")
            p_closest = min(put_strikes, key=lambda x: abs(x - S)) if put_strikes else 0
            if "insp_put_k" in st.session_state and st.session_state["insp_put_k"] not in put_strikes:
                del st.session_state["insp_put_k"]
            if "insp_put_k" not in st.session_state:
                st.session_state["insp_put_k"] = p_closest

            p_strike = st.selectbox("Select Put Strike", put_strikes, key="insp_put_k")
            p_row = puts[puts['strike'] == p_strike].iloc[0]

            p_bid, p_ask = p_row.get("bid", 0.0), p_row.get("ask", 0.0)
            p_mid = (p_bid + p_ask) / 2 if (p_bid > 0 and p_ask > 0) else p_row['lastPrice']
            p_iv = implied_vol(p_mid, S, p_strike, T, r, q, 'put')

            p1, p2, p4 = st.columns(3)
            p1.metric("Bid", f"${p_bid:.2f}")
            p2.metric("Ask", f"${p_ask:.2f}")
            p4.metric("Market Implied Vol (Mid)", f"{p_iv*100:.2f}%" if p_iv else "N/A")

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

        for leg in legs:
            if leg["qty"] == 0: continue
            df = calls if leg["type"] == "Call" else puts
            row = df[df['strike'] == leg["k"]]
            if row.empty: continue

            active_strikes.append(leg["k"])
            bid, ask = row.iloc[0].get("bid", 0.0), row.iloc[0].get("ask", 0.0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else row.iloc[0].get("lastPrice", 0.0)
            real_price = bid if leg["pos"] == "Short" else ask

            iv_mid = implied_vol(mid, S, leg["k"], T, r, q, leg["type"].lower())
            iv_real = implied_vol(real_price, S, leg["k"], T, r, q, leg["type"].lower())

            leg_delta_mid = calculate_delta(S, leg["k"], T, r, q, iv_mid, leg["type"].lower())
            leg_delta_real = calculate_delta(S, leg["k"], T, r, q, iv_real, leg["type"].lower())

            sign = 1 if leg["pos"] == "Long" else -1

            net_cf += (real_price * -sign) * leg["qty"]
            net_delta_mid += (leg_delta_mid * sign) * leg["qty"]
            net_delta_real += (leg_delta_real * sign) * leg["qty"]

        hedge_shares_mid = -net_delta_mid
        hedge_shares_real = -net_delta_real

        def calc_options_payoff(s_arr):
            total_p = np.zeros_like(s_arr)
            for l in legs:
                if l["qty"] == 0: continue
                df = calls if l["type"] == "Call" else puts
                row = df[df['strike'] == l["k"]]
                if row.empty: continue
                entry_price = row.iloc[0]['bid'] if l["pos"] == "Short" else row.iloc[0]['ask']
                sign = 1 if l["pos"] == "Long" else -1
                v = np.maximum(s_arr - l["k"], 0) if l["type"] == "Call" else np.maximum(l["k"] - s_arr, 0)
                total_p += (v - entry_price) * sign * l["qty"]
            return total_p

        def calc_hedged_payoff(s_arr, options_payoff, hedge_shares):
            return options_payoff + (hedge_shares * (s_arr - S))

        crit_points = active_strikes + [S]
        s_min_c = min(crit_points) * 0.75 if crit_points else S * 0.75
        s_max_c = max(crit_points) * 1.25 if crit_points else S * 1.25
        test_s = np.linspace(s_min_c, s_max_c, 5000)

        if sum(l["qty"] for l in legs) > 0:
            unhedged_payoff = calc_options_payoff(test_s)
            hedged_payoff_mid = calc_hedged_payoff(test_s, unhedged_payoff, hedge_shares_mid)
            hedged_payoff_real = calc_hedged_payoff(test_s, unhedged_payoff, hedge_shares_real)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("##### Hedging & Delta Summary")
            st.caption("Compare the theoretical assumptions of Mid-Price hedging against Real-Price (Bid/Ask) hedging.")

            c_sum1, c_sum2, c_sum3, c_sum4, c_sum5 = st.columns(5)
            c_sum1.metric("Net Premium Collected", f"${net_cf:.2f}")
            c_sum2.metric("Net Delta (Mid)", f"{net_delta_mid:.4f}")
            c_sum3.metric("Net Delta (Real)", f"{net_delta_real:.4f}")
            c_sum4.metric("Hedge Shares (Mid)", f"{hedge_shares_mid:.4f}")
            c_sum5.metric("Hedge Shares (Real)", f"{hedge_shares_real:.4f}")
            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("#### Combined Strategy Payoff")
            st.caption("💡 **Tip:** Click on the labels in the legend on the right to show/hide specific payoff lines.")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[s_min_c, s_max_c], y=[0, 0], name='Zero Line', line=dict(color='gray', dash='dash'), hoverinfo='skip'))

            fig2.add_trace(go.Scatter(x=test_s, y=unhedged_payoff, name='Unhedged Options', line=dict(color='white', width=3), legendgroup='unhedged'))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(unhedged_payoff >= 0, unhedged_payoff, 0), fill='tozeroy', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(width=0), name="Profit (Unhedged)", legendgroup='unhedged', showlegend=False))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(unhedged_payoff < 0, unhedged_payoff, 0), fill='tozeroy', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(width=0), name="Loss (Unhedged)", legendgroup='unhedged', showlegend=False))

            fig2.add_trace(go.Scatter(x=test_s, y=hedged_payoff_mid, name='Hedged (Mid Price)', line=dict(color='#00BFFF', width=3, dash='dash'), legendgroup='mid'))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(hedged_payoff_mid >= 0, hedged_payoff_mid, 0), fill='tozeroy', fillcolor='rgba(0, 191, 255, 0.1)', line=dict(width=0), name="Profit (Mid)", legendgroup='mid', showlegend=False))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(hedged_payoff_mid < 0, hedged_payoff_mid, 0), fill='tozeroy', fillcolor='rgba(0, 191, 255, 0.05)', line=dict(width=0), name="Loss (Mid)", legendgroup='mid', showlegend=False))

            fig2.add_trace(go.Scatter(x=test_s, y=hedged_payoff_real, name='Hedged (Real Price)', line=dict(color='#00E676', width=3), legendgroup='real'))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(hedged_payoff_real >= 0, hedged_payoff_real, 0), fill='tozeroy', fillcolor='rgba(0, 255, 100, 0.15)', line=dict(width=0), name="Profit (Real)", legendgroup='real', showlegend=False))
            fig2.add_trace(go.Scatter(x=test_s, y=np.where(hedged_payoff_real < 0, hedged_payoff_real, 0), fill='tozeroy', fillcolor='rgba(255, 50, 50, 0.15)', line=dict(width=0), name="Loss (Real)", legendgroup='real', showlegend=False))

            for k in active_strikes:
                fig2.add_vline(x=k, line_dash="dot", line_color="rgba(255, 255, 255, 0.4)", annotation_text=f"K={k}")

            fig2.update_layout(
                height=600,
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Set at least one Leg's Quantity > 0 to see the Strategy Plots.")
    else:
        st.error("No options data available for this ticker.")
