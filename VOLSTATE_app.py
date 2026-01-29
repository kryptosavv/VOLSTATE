import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- CONFIG ---
DB_NAME = "fake_market_data.db"

st.set_page_config(
    page_title="VOLSTATE Dashboard", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { font-family: 'Segoe UI', sans-serif; background-color: #0e1117; }
    .top-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
    .pill-container { display: flex; gap: 10px; margin-bottom: 15px; }
    .pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #fff; display: inline-block; }
    .pill-yellow { background-color: #ffc107; color: #000; }
    .pill-red { background-color: #dc3545; }
    .pill-orange { background-color: #fd7e14; }
    .pill-gray { background-color: #444; opacity: 0.5; }
    
    /* Subtle Highlight for RPV Box */
    .regime-box { 
        text-align: center; 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 25px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .regime-label { font-size: 42px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
    .regime-sub { font-size: 14px; opacity: 0.8; margin-top: 5px; font-family: monospace; }
    .grid-tile { background-color: #161b22; border: 1px solid #333; border-radius: 6px; padding: 15px; height: 110px; display: flex; flex-direction: column; justify-content: space-between; }
    .tile-header { font-size: 13px; color: #888; text-transform: uppercase; font-weight: 600; }
    .tile-value { font-size: 24px; font-weight: 800; margin: 2px 0; }
    .tile-sub { font-size: 12px; font-family: monospace; color: #aaa; }
    .strat-box { background-color: #1e252e; border-left: 4px solid #555; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }
    .strat-header { font-size: 16px; font-weight: bold; color: #ddd; margin-bottom: 10px; text-transform: uppercase; }
    .strat-list { margin-bottom: 0; padding-left: 20px; color: #bbb; font-size: 14px; }
    .strat-list li { margin-bottom: 5px; }
    .text-green { color: #28a745; }
    .text-amber { color: #ffc107; }
    .text-red { color: #dc3545; }
    .text-gray { color: #888; }
    .border-green { border-left: 4px solid #28a745; }
    .border-amber { border-left: 4px solid #ffc107; }
    .border-red { border-left: 4px solid #dc3545; }
    .border-gray { border-left: 4px solid #555; }
    .mini-diag { font-family: monospace; font-size: 12px; color: #666; border-top: 1px solid #333; margin-top: 20px; padding-top: 10px; display: flex; justify-content: space-around; }
    .section-header { margin-top: 40px; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #333; font-size: 20px; font-weight: bold; color: #ddd;}
    .rpv-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 10px; width: 100%; }
    .rpv-seg { height: 100%; }
    /* Hide Streamlit Date Input Label if desired */
    .stDateInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADERS ---
def load_data(limit=300):
    conn = sqlite3.connect(DB_NAME)
    query = f"""
        SELECT * FROM market_logs 
        ORDER BY timestamp DESC LIMIT {limit}
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cols_to_numeric = ['m1_iv', 'm3_iv', 'm1_straddle', 'spot_price', 'm2_iv', 'skew_index', 'india_vix']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df

# --- UI HELPER: RENDER TILE ---
def render_tile(label, state_bool, display_text, subtext, is_stress=False):
    if state_bool is None:
        color_class = "text-gray"
        border_class = "border-gray"
    elif state_bool:
        color_class = "text-red" if is_stress else "text-amber"
        border_class = "border-red" if is_stress else "border-amber"
    else:
        color_class = "text-green"
        border_class = "border-green"
        
    st.markdown(f"""
    <div class="grid-tile {border_class}">
        <div class="tile-header">{label}</div>
        <div class="tile-value {color_class}">{display_text}</div>
        <div class="tile-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 1️⃣ SIGNAL LIKELIHOOD FUNCTIONS (RPV CORE) ---

def iv_likelihood(iv_chg):
    return {
        "COMPRESSION": max(0, 1 - iv_chg/0.5),
        "TRANSITION": np.clip(iv_chg/0.6, 0, 1),
        "EXPANSION": np.clip(iv_chg/1.0, 0, 1),
        "STRESS": np.clip((iv_chg - 0.8)/1.2, 0, 1)
    }

def straddle_likelihood(std_pct):
    return {
        "COMPRESSION": 1 if std_pct < -0.2 else 0.2,
        "TRANSITION": np.clip((std_pct + 0.2)/0.4, 0, 1),
        "EXPANSION": np.clip((std_pct + 0.1)/0.6, 0, 1),
        "STRESS": np.clip((std_pct - 0.8)/1.0, 0, 1)
    }

def back_month_likelihood(bm_spread):
    return {
        "COMPRESSION": np.clip(-bm_spread / 0.3, 0, 1),
        "TRANSITION": np.clip(bm_spread/0.4, 0, 1),
        "EXPANSION": np.clip(bm_spread/0.6, 0, 1),
        "STRESS": np.clip(bm_spread/0.8, 0, 1)
    }

def term_likelihood(slope):
    return {
        "COMPRESSION": np.clip(slope/1.5, 0, 1),
        "TRANSITION": np.clip((1.2 - slope)/1.2, 0, 1),
        "EXPANSION": np.clip((0.8 - slope)/0.8, 0, 1),
        "STRESS": np.clip((-slope - 0.3)/0.7, 0, 1)
    }

def skew_likelihood(skew_chg):
    return {
        "COMPRESSION": 1 if skew_chg <= 0 else 0,
        "TRANSITION": np.clip((0.3 - skew_chg)/0.3, 0, 1),
        "EXPANSION": np.clip(skew_chg/0.6, 0, 1),
        "STRESS": np.clip((skew_chg - 0.6)/1.0, 0, 1)
    }

def disconnect_likelihood(disconnect):
    return {
        "COMPRESSION": 0,
        "TRANSITION": 0.4 if disconnect else 0.6,
        "EXPANSION": 0.7 if disconnect else 0.4,
        "STRESS": 0.9 if disconnect else 0.2
    }

# --- 2️⃣ COMPUTE REGIME PROBABILITY VECTOR (RPV) ---

REGIMES = ["COMPRESSION", "TRANSITION", "EXPANSION", "STRESS"]
WEIGHTS = {
    "iv": 1.2, "straddle": 1.2, "back_month": 1.0, 
    "term": 1.0, "skew": 1.4, "disconnect": 0.8
}

def compute_rpv(curr, prev, prev2):
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    m2_curr, m2_prev = curr.get('m2_iv', 0), prev.get('m2_iv', 0)
    bm_spread = (m2_curr - m2_prev) - iv_chg
    slope = curr['m3_iv'] - curr['m1_iv']
    skew_chg = curr['skew_index'] - prev['skew_index']
    spot_chg = ((curr['spot_price'] - prev['spot_price']) / prev['spot_price']) * 100
    is_disconnect = (abs(spot_chg) < 0.1 and iv_chg > 0.5)

    lhs = {
        "iv": iv_likelihood(iv_chg),
        "straddle": straddle_likelihood(std_pct),
        "back_month": back_month_likelihood(bm_spread),
        "term": term_likelihood(slope),
        "skew": skew_likelihood(skew_chg),
        "disconnect": disconnect_likelihood(is_disconnect)
    }

    scores = {r: 0.0 for r in REGIMES}
    for sig, lk_map in lhs.items():
        for r in REGIMES:
            scores[r] += WEIGHTS[sig] * lk_map[r]
            
    total = sum(scores.values())
    rpv = {r: (scores[r]/total if total > 0 else 0) for r in REGIMES}
    return rpv, lhs

# --- 3️⃣ RISK POSTURE & ENTROPY ---

def regime_entropy(rpv):
    return -sum(p * np.log(p + 1e-9) for p in rpv.values())

def derive_risk_posture(rpv):
    if rpv["STRESS"] > 0.20:
        hedge_status = ("MANDATORY", "#dc3545") # Red
    elif rpv["STRESS"] > 0.15:
        hedge_status = ("ACCUMULATE", "#ffc107") # Yellow/Amber
    else:
        hedge_status = ("OPTIONAL", "#666")      # Grey

    return {
        "long_gamma": (rpv["EXPANSION"] + rpv["STRESS"] > 0.5) and (rpv["COMPRESSION"] < 0.3),
        "short_theta": rpv["COMPRESSION"] > 0.5,
        "tail_hedge_data": hedge_status, 
        "carry_allowed": rpv["COMPRESSION"] > 0.4 and rpv["STRESS"] < 0.15
    }

# --- 4️⃣ REGIME DYNAMICS LOGIC ---

def compute_rpv_series(df):
    df = df.sort_values('timestamp').copy()
    rows = []
    if len(df) < 3: return pd.DataFrame()
    for i in range(2, len(df)):
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rpv, _ = compute_rpv(curr, prev, prev2)
        rpv['timestamp'] = curr['timestamp']
        rows.append(rpv)
    return pd.DataFrame(rows)

def compute_rpv_drift(rpv_df, lookback=3):
    drift = {}
    for r in REGIMES:
        if len(rpv_df) > lookback:
            drift[r] = rpv_df[r].iloc[-1] - rpv_df[r].iloc[-(lookback+1)]
        else:
            drift[r] = 0.0
    return drift

def detect_pre_stress(rpv_df):
    if len(rpv_df) < 4: return False, {}
    stress = rpv_df["STRESS"].values
    expansion = rpv_df["EXPANSION"].values
    slope = stress[-1] - stress[-3]
    accel = (stress[-1] - stress[-2]) > (stress[-2] - stress[-3])
    
    conditions = {
        "stress_level": stress[-1] > 0.20,
        "stress_slope": slope > 0.08,
        "stress_accel": accel,
        "stress_vs_exp": stress[-1] > expansion[-1] * 0.6
    }
    return all(conditions.values()), conditions

# --- ENGINE WRAPPERS ---

SIGNAL_LABELS = {
    "iv": "ATM IV rising", "straddle": "Straddle decay stalled",
    "back_month": "Back-month IV bidding", "term": "Term structure stress",
    "skew": "Skew expanding (tail demand)", "disconnect": "Spot–Vol disconnect"
}

def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 3: return None, None, df_c.iloc[-1]

    curr, prev, prev2 = df_c.iloc[-1], df_c.iloc[-2], df_c.iloc[-3]
    rpv, lhs = compute_rpv(curr, prev, prev2)
    dominant = max(rpv, key=rpv.get)
    
    colors = {"COMPRESSION": "#28a745", "TRANSITION": "#ffc107", "EXPANSION": "#fd7e14", "STRESS": "#dc3545"}
    color = colors.get(dominant, "#888")
    
    entropy = regime_entropy(rpv)
    if rpv[dominant] > 0.55 and entropy < 1.1: confidence = "HIGH"
    elif rpv[dominant] > 0.4: confidence = "MEDIUM"
    else: confidence = "LOW (UNSTABLE)"

    risk_posture = derive_risk_posture(rpv)
    drivers = [SIGNAL_LABELS[k] for k, v in lhs.items() if v[dominant] > 0.6]
    counterforces = [SIGNAL_LABELS[k] for k, v in lhs.items() if v[dominant] < 0.3]
    
    dte = curr.get('m1_dte', 30)
    is_roll, is_late = dte >= 28, dte <= 7
    
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    slope = curr['m3_iv'] - curr['m1_iv']
    skew_chg = curr['skew_index'] - prev['skew_index']
    
    signals = {
        't1': (iv_chg > 0.2, "RISING" if iv_chg > 0.2 else "STABLE", f"{iv_chg:+.2f}%"),
        't2': (std_pct > -0.1, "STALLED" if std_pct > -0.1 else "DECAYING", f"{std_pct:+.2f}%"),
        't3': (None, "SEE RPV", "Probabilistic"),
        't4': (slope < 0, "INVERTED" if slope < 0 else "NORMAL", f"{slope:.2f}"),
        't5': (skew_chg > 0.3, "RISING" if skew_chg > 0.3 else "FLAT", f"{skew_chg:+.2f}"),
        't6': (None, "SEE RPV", "Probabilistic")
    }

    context = {
        'regime': dominant, 'color': color, 'confidence': confidence,
        'rpv': rpv, 'risk': risk_posture, 'drivers': drivers, 'counterforces': counterforces,
        'is_roll': is_roll, 'is_late': is_late, 'dte': dte
    }
    return signals, context, curr

def calculate_historical_regime(df):
    df_sorted = df.sort_values('timestamp', ascending=True).copy()
    history = []
    for i in range(2, len(df_sorted)):
        curr, prev, prev2 = df_sorted.iloc[i], df_sorted.iloc[i-1], df_sorted.iloc[i-2]
        rpv, _ = compute_rpv(curr, prev, prev2)
        dom = max(rpv, key=rpv.get)
        history.append({'timestamp': curr['timestamp'], 'regime': dom, 'val': 1})
    return pd.DataFrame(history)

def get_full_rpv_history(df):
    df = df.sort_values('timestamp').copy()
    rows = []
    if len(df) < 3: return pd.DataFrame()
    for i in range(2, len(df)):
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        rpv, _ = compute_rpv(curr, prev, prev2)
        row = rpv.copy()
        row['timestamp'] = curr['timestamp']
        rows.append(row)
    return pd.DataFrame(rows)

# --- PAGE RENDERING FUNCTIONS ---

def render_dashboard_key():
    with st.expander("ℹ️ **Dashboard Key & Glossary (Click to Expand)**", expanded=False):
        st.markdown("""
        ### 📊 **1. The Regime Banner (RPV)**
        * **Regime Probability Vector (RPV):** The multi-colored bar shows the "mix" of market states. Markets are rarely 100% one thing.
        * **Confidence:** Based on *Entropy*. "High" means the market has a clear direction. "Low" means signals are conflicting.
        
        ### 🧭 **2. Regime Dynamics (Momentum)**
        * **📈 RPV Drift:** Measures the *speed* of change. If Stress is low but rising fast (Green Arrow), be careful.
        * **⚠️ Pre-Stress Monitor:** Detects hidden acceleration in tail risk before the VIX spikes.

        ### 📉 **3. Analytics Charts (Detailed Breakdown)**
        
        **1. Regime Timeline (Dots)**
        * *What:* Shows the historical dominance of regimes over time.
        * *Interpret:* Look for stability (clustered dots) vs instability (rapid color switching). 

        **2. Nifty Spot vs ATM Straddle**
        * *What:* Compares Nifty price (Blue) to Straddle Price (Red Dotted).
        * *Bullish (for Sellers):* Red line sloping down. Straddle is losing value (Theta decay is working).
        * *Bearish (for Sellers):* Red line spiking up. Movement is outpacing expected decay (Gamma risk).

        **3. Regime Probabilities (Stacked Area)**
        * *What:* The "under the hood" view of the RPV evolution.
        * *Interpret:* Watch the **Red (Stress)** and **Orange (Expansion)** layers. If these areas are growing wider over time, structural risk is increasing.

        **4. Term Structure Slope**
        * *What:* Difference between Far Month IV and Near Month IV.
        * *Green (Positive):* Contango. Normal market. 
        * *Red (Negative):* Backwardation. Inverted curve indicating panic.

        **5. Daily Straddle Change %**
        * *What:* The daily P&L of holding an ATM Straddle.
        * *Green (Negative):* The straddle lost value (Good for short volatility).
        * *Red (Positive):* The straddle gained value (Good for long volatility).

        **6. VRP Index (Edge)**
        * *What:* Volatility Risk Premium (Implied Vol - Realized Vol).
        * *Green:* IV is expensive relative to actual movement (Edge for sellers).
        * *Red:* IV is cheap relative to actual movement (Edge for buyers).

        **7. Skew Index**
        * *What:* The relative cost of Puts vs Calls.
        * *Rising:* Institutions are buying Puts (Fear/Hedging).
        * *Falling:* Complacency (Greed).

        **8. India VIX**
        * *What:* The baseline fear gauge for the Indian market.
        * *Interpret:* Use as a confirmation tool, not a leading indicator.

        **9. Price Displacement (SD)**
        * *What:* Price movement measured in Standard Deviations (Sigma).
        * *Red Dashed Line:* The "1 Sigma" limit. Moves above this line are statistically significant and break the "Compression" regime.
        """, unsafe_allow_html=True)

def render_thesis_section():
    with st.expander("📘 **Methodology & Thesis (Deep Dive)**", expanded=False):
        st.markdown("""
        ### 🧮 1. The Core Philosophy: Fuzzy Logic
        Markets rarely flip a binary switch from "Safe" to "Dangerous." They drift through gradients of probability. 
        Instead of asking *"Is the market in Stress?"* (Yes/No), we ask: 
        > *"What is the probability that we are in Stress versus Transition?"*

        This engine calculates a **Regime Probability Vector (RPV)** for every data point:
        * **Binary:** `Stress = False`
        * **VOLSTATE:** `Stress = 0.22`, `Transition = 0.45`, `Expansion = 0.33`

        ---

        ### 📡 2. The Inputs (Sensors)
        We use 6 "Vital Signs" to diagnose the market. Each input is chosen to measure a specific structural characteristic of volatility.

        #### 🌡️ 1. ATM IV Momentum (`Weight: 1.2`)
        * **Question:** Is the "price of anxiety" rising or falling?
        * **The Logic:** Volatility exhibits mean reversion.
            * **Rising IV:** Markets are pricing in *new* risks (Stress/Transition).
            * **Falling IV:** Markets are digesting *known* risks (Compression).
        
        #### ⏳ 2. Straddle Decay (`Weight: 1.2`)
        * **Question:** Is the market moving enough to pay the rent?
        * **The Logic:** An ATM Straddle (Call + Put) bleeds value daily due to Time Decay (Theta).
            * **Decaying:** The market is pinned. Sellers are winning. (Compression).
            * **Holding/Rising:** Price movement is outpacing time decay. (Expansion/Stress).

        #### 🔭 3. Back-Month Spread (`Weight: 1.0`)
        * **Question:** Is there "stealth" bidding for future protection?
        * **The Logic:** Smart money often hedges 30-60 days out before a storm.
            * **Spread Widening:** Traders fear the future more than the present. A **Transition** is starting.

        #### 📉 4. Term Structure Slope (`Weight: 1.0`)
        * **Question:** Is the curve Healthy (Contango) or Panic (Inverted)?
        * **The Logic:**
            * **Contango (Slope > 0):** Normal. Future is uncertain, so long-dated insurance costs more.
            * **Backwardation (Slope < 0):** Panic. Demand for *immediate* liquidity overrides future value. This is the hallmark of **Stress**.

        #### ⚠️ 5. Skew Index (`Weight: 1.4`)
        * **Question:** Are Puts getting expensive relative to Calls?
        * **The Logic:** Institutions buy Puts to hedge. Retail buys Calls to speculate.
            * **Rising Skew:** "Smart money" is hedging tail risk, even if the market is rising. This is the most reliable leading indicator of fragility.
        
        #### 🔗 6. Spot-Vol Disconnect (`Weight: 0.8`)
        * **Question:** Is the physics of the market broken?
        * **The Logic:** Normally, Price UP = Volatility DOWN.
            * **Disconnect:** If Price rises AND Volatility rises, it indicates instability (Euphoria or Short Squeeze). It typically precedes a violent reversal.

        ---

        ### 🧠 3. The Logic Engine
        
        **The Aggregation**
        The system takes the 6 sensor inputs, applies their respective weights, and normalizes the result to sum to 100%. This creates the **RPV**.

        **The "BS Detector" (Entropy)**
        We calculate the **Entropy** of the probability vector to measure confusion.
        * **Low Entropy:** The market has picked a clear direction (e.g., 90% Compression).
        * **High Entropy:** The signals are conflicting (e.g., 25% across all four). The dashboard labels this **"LOW CONFIDENCE (UNSTABLE)"**.

        **Risk Posture (Translation to Action)**
        The probability scores are translated into trading rules:
        
        1.  **Tail Hedge Logic:**
            * **Stress < 15%:** `OPTIONAL` (Ignore tail risk).
            * **Stress 15-20%:** `ACCUMULATE` (Buy cheap OTM puts. Momentum is building).
            * **Stress > 20%:** `MANDATORY` (Capital preservation mode. Volatility is now expensive).
        
        2.  **Long Gamma Logic:**
            * We only buy options when `(Expansion + Stress > 50%)` **AND** `(Compression < 30%)`. This prevents buying breakouts in "fake" markets.
        """)

def render_dashboard(df_selected, signals, ctx, curr, df_all):
    # --- REGIME BANNER (HIGHLIGHTED & UPDATED) ---
    rpv_html = f"""<div class="regime-box" style="background-color: {ctx['color']}15; border-color: {ctx['color']}80;">
<div style="text-align: left; font-size: 11px; color: #888; letter-spacing: 1px; font-weight: 600;">REGIME PROBABILITY VECTOR (RPV)</div>
<div class="regime-sub" style="color: {ctx['color']}; margin-top: 5px;">DETECTED REGIME (PROBABILISTIC)</div>
<div class="regime-label" style="color: {ctx['color']}">{ctx['regime']}</div>
<div class="regime-sub" style="margin-top: 5px;">CONFIDENCE: {ctx['confidence']}</div>

<div class="rpv-bar" style="display: flex; height: 8px; background: #222; border-radius: 4px; overflow: hidden; margin-top: 15px; width: 100%;">
    <div style="width: {ctx['rpv']['COMPRESSION']*100}%; background: #28a745;" title="Compression"></div>
    <div style="width: {ctx['rpv']['TRANSITION']*100}%; background: #ffc107;" title="Transition"></div>
    <div style="width: {ctx['rpv']['EXPANSION']*100}%; background: #fd7e14;" title="Expansion"></div>
    <div style="width: {ctx['rpv']['STRESS']*100}%; background: #dc3545;" title="Stress"></div>
</div>

<div style="display: flex; justify-content: space-between; font-size: 11px; color: #bbb; margin-top: 8px; width: 100%;">
    <div style="text-align: center; flex: 1; border-right: 1px solid #333;">COMP<br><span style="color: #28a745; font-weight: bold;">{ctx['rpv']['COMPRESSION']:.0%}</span></div>
    <div style="text-align: center; flex: 1; border-right: 1px solid #333;">TRANS<br><span style="color: #ffc107; font-weight: bold;">{ctx['rpv']['TRANSITION']:.0%}</span></div>
    <div style="text-align: center; flex: 1; border-right: 1px solid #333;">EXP<br><span style="color: #fd7e14; font-weight: bold;">{ctx['rpv']['EXPANSION']:.0%}</span></div>
    <div style="text-align: center; flex: 1;">STRS<br><span style="color: #dc3545; font-weight: bold;">{ctx['rpv']['STRESS']:.0%}</span></div>
</div>
</div>"""
    st.markdown(rpv_html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: s = signals['t1']; render_tile("ATM IV MOMENTUM", s[0], s[1], s[2])
    with c2: s = signals['t2']; render_tile("STRADDLE DECAY", s[0], s[1], s[2])
    with c3: s = signals['t3']; render_tile("BACK MONTH BID", s[0], s[1], s[2])

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4: s = signals['t4']; render_tile("TERM STRUCTURE", s[0], s[1], s[2], is_stress=True)
    with c5: s = signals['t5']; render_tile("TAIL RISK (SKEW)", s[0], s[1], s[2])
    with c6: s = signals['t6']; render_tile("VOL vs SPOT DISCONNECT", s[0], s[1], s[2])

    st.markdown(f"""
        <div class="mini-diag">
            <span>SPOT: {curr['spot_price']:.0f}</span>
            <span>ATM IV: {curr['m1_iv']:.2f}%</span>
            <span>STRADDLE: {curr['m1_straddle']:.0f}</span>
            <span>DTE: {ctx['dte']}</span>
        </div>
    """, unsafe_allow_html=True)

    rp = ctx['risk']
    drivers_str = ", ".join(ctx['drivers']) if ctx['drivers'] else "None"
    counter_str = ", ".join(ctx['counterforces']) if ctx['counterforces'] else "None"
    
    st.markdown(f"""
    <div class="strat-box" style="border-left-color: {ctx['color']};">
        <div class="strat-header" style="color: {ctx['color']}">🧭 Risk Posture & Drivers</div>
        <div style="display: flex; gap: 40px;">
            <div style="flex: 1;">
                <strong style="color: #ddd;">✅ REGIME DRIVERS:</strong>
                <div style="color: #bbb; margin-top: 5px; font-size: 13px;">{drivers_str}</div>
                <br>
                <strong style="color: #888;">⛔ COUNTERFORCES:</strong>
                <div style="color: #666; margin-top: 5px; font-size: 13px;">{counter_str}</div>
            </div>
            <div style="flex: 1;">
                <strong style="color: #ddd;">🛡️ POSTURE:</strong>
                <ul class="strat-list">
                    <li>Long Gamma: <span style="color: {'#28a745' if rp['long_gamma'] else '#666'}">{"YES" if rp['long_gamma'] else "NO"}</span></li>
                    <li>Short Theta: <span style="color: {'#28a745' if rp['short_theta'] else '#666'}">{"ALLOWED" if rp['short_theta'] else "REDUCE"}</span></li>
                    <li>Tail Hedge: <span style="color: {rp['tail_hedge_data'][1]}">{rp['tail_hedge_data'][0]}</span></li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- REGIME DYNAMICS ---
    st.markdown('<div class="section-header">🧭 Regime Dynamics</div>', unsafe_allow_html=True)
    
    rpv_df_hist = compute_rpv_series(df_selected.tail(20))
    drift = compute_rpv_drift(rpv_df_hist)
    pre_stress, ps_details = detect_pre_stress(rpv_df_hist)
    
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("##### 📈 RPV Drift (Momentum)")
        cols = st.columns(4)
        for i, r in enumerate(REGIMES):
            arrow = "↑" if drift[r] > 0.01 else "↓" if drift[r] < -0.01 else "-"
            color = "green" if drift[r] > 0 else "red" if drift[r] < 0 else "gray"
            with cols[i]:
                st.metric(label=r.title(), value=f"{drift[r]:.2f}", delta=arrow)
    
    with dc2:
        st.markdown("##### ⚠️ Pre-Stress Monitor")
        if pre_stress:
            st.error("🚨 TAIL RISK BUILDING (Acceleration Detected)")
            st.write(f"Slope: {ps_details['stress_slope']:.2f} | Accel: {ps_details['stress_accel']}")
        else:
            st.success("✅ No imminent tail acceleration detected.")
            last_stress = rpv_df_hist['STRESS'].iloc[-1] if not rpv_df_hist.empty else 0.0
            st.caption(f"Stress Level: {last_stress:.2f} (Threshold 0.20)")

    # --- CHARTS ---
    st.markdown('<div class="section-header">📊 History (Analytics)</div>', unsafe_allow_html=True)
    
    df_chart = df_selected.sort_values('timestamp', ascending=True).tail(60).copy()
    
    # 1. Regime Timeline
    df_regime = calculate_historical_regime(df_chart)
    if not df_regime.empty:
        df_regime['y_val'] = 1
        fig_regime = px.scatter(df_regime, x="timestamp", y="y_val", color="regime", 
                            color_discrete_map={
                                "COMPRESSION": "#28a745", "TRANSITION": "#ffc107", 
                                "EXPANSION": "#fd7e14", "STRESS": "#dc3545"
                            },
                            symbol_sequence=['square'], title="<b>Regime Timeline</b>")
        fig_regime.update_traces(marker=dict(size=15))
        fig_regime.update_layout(template="plotly_dark", height=130, showlegend=False, 
                               yaxis=dict(visible=False), xaxis=dict(showgrid=False), margin=dict(t=30, b=10))
        st.plotly_chart(fig_regime, width="stretch")

    df_hist = df_chart.copy()
    df_hist['slope'] = df_hist['m3_iv'] - df_hist['m1_iv']
    df_hist['slope_col'] = np.where(df_hist['slope'] >= 0, '#00cc00', '#ff0000')
    df_hist['std_pct'] = df_hist['m1_straddle'].pct_change() * 100
    df_hist['std_col'] = np.where(df_hist['std_pct'] <= 0, '#00cc00', '#ff0000') 
    df_hist['log_ret'] = np.log(df_hist['spot_price'] / df_hist['spot_price'].shift(1))
    df_hist['rv_5d'] = df_hist['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
    df_hist['vrp'] = df_hist['m1_iv'] - df_hist['rv_5d']
    df_hist['vrp_col'] = np.where(df_hist['vrp'] > 0, '#00cc00', '#ff0000') 
    
    # 2. Spot vs Straddle
    fig_spot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spot.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['spot_price'], line=dict(color='#3498db', width=2), name="Nifty Spot"), secondary_y=False)
    fig_spot.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['m1_straddle'], line=dict(color='#e74c3c', width=2, dash='dot'), name="ATM Straddle"), secondary_y=True)
    fig_spot.update_layout(title="<b>Nifty Spot vs ATM Straddle Price Trend</b>", template="plotly_dark", height=350, margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_spot, width="stretch")

    # 3. Regime Probabilities Stacked Area
    rpv_full_hist = get_full_rpv_history(df_selected.tail(60))
    if not rpv_full_hist.empty:
        fig_rpv = go.Figure()
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['STRESS'], mode='lines', stackgroup='one', name='STRESS', line=dict(color='#dc3545', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['EXPANSION'], mode='lines', stackgroup='one', name='EXPANSION', line=dict(color='#fd7e14', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['TRANSITION'], mode='lines', stackgroup='one', name='TRANSITION', line=dict(color='#ffc107', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['COMPRESSION'], mode='lines', stackgroup='one', name='COMPRESSION', line=dict(color='#28a745', width=0)))
        
        fig_rpv.update_layout(title="<b>Regime Probabilities Over Time (2 Months)</b>", template="plotly_dark", height=300, margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(range=[0, 1]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_rpv, width="stretch")

    col_r1_1, col_r1_2 = st.columns(2)
    with col_r1_1:
        fig_slope = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['slope'], marker_color=df_hist['slope_col']))
        fig_slope.update_layout(title="<b>Term Structure Slope</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_slope, width="stretch")
    with col_r1_2:
        fig_std = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['std_pct'], marker_color=df_hist['std_col']))
        fig_std.update_layout(title="<b>Daily Straddle Change %</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_std, width="stretch")

    col_r2_1, col_r2_2 = st.columns(2)
    with col_r2_1:
        fig_vrp = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['vrp'], marker_color=df_hist['vrp_col']))
        fig_vrp.update_layout(title="<b>VRP Index (Edge)</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vrp, width="stretch")
    with col_r2_2:
        fig_skew = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['skew_index'], mode='lines', line=dict(color='#3498db', width=2), fill='tozeroy'))
        fig_skew.update_layout(title="<b>Skew Index</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_skew, width="stretch")

    col_r3_1, col_r3_2 = st.columns(2)
    with col_r3_1:
        fig_vvix = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['india_vix'], mode='lines', line=dict(color='#f1c40f', width=2)))
        fig_vvix.update_layout(title="<b>INDIA VIX</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vvix, width="stretch")
    with col_r3_2:
        fig_sd = go.Figure()
        daily_iv = (df_hist['m1_iv'] / 100) / np.sqrt(252)
        spot_pct = df_hist['spot_price'].pct_change()
        df_hist['sd_move'] = (spot_pct / daily_iv.shift(1)).abs().fillna(0)
        
        fig_sd.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['sd_move'], fill='tozeroy', mode='lines', line=dict(color='#9b59b6')))
        fig_sd.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig_sd.update_layout(title="<b>Price Displacement (SD)</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_sd, width="stretch")

def main():
    df_all = load_data(300) 
    if len(df_all) < 5: st.error("⚠️ Not enough data found."); st.stop()

    # --- TOP BAR ---
    c_head1, c_head2, c_head3 = st.columns([1, 2, 1])
    max_date = df_all['timestamp'].max().date()
    min_date = df_all['timestamp'].min().date()

    with c_head1:
        st.markdown(f"**{df_all.iloc[0]['timestamp'].strftime('%d %b %Y | %H:%M')} IST**")
        selected_date = st.date_input("Go to Date:", value=max_date, min_value=min_date, max_value=max_date)

    df_selected = df_all[df_all['timestamp'].dt.date <= selected_date]
    if df_selected.empty: st.stop()

    signals, ctx, curr = run_engine_live(df_selected)

    with c_head2:
        st.markdown("""<h1 style='text-align: center; margin: 0; padding: 0; color: #ffc107; font-size: 42px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0px 0px 15px rgba(255, 193, 7, 0.4);'>⚡ VOLSTATE DASHBOARD</h1>""", unsafe_allow_html=True)
        
    with c_head3:
        st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
        if ctx['is_roll']: st.markdown('<span class="pill pill-yellow">🟡 ROLLOVER WINDOW</span>', unsafe_allow_html=True)
        elif ctx['is_late']: st.markdown('<span class="pill pill-orange">🟠 LATE CYCLE</span>', unsafe_allow_html=True)
        else: st.markdown('<span class="pill pill-gray">MID CYCLE</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- RENDER DASHBOARD ---
    render_dashboard(df_selected, signals, ctx, curr, df_all)
    
    # --- FOOTER SECTIONS ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    # 1. Dashboard Key & Glossary
    render_dashboard_key()
    
    # 2. Methodology & Thesis
    render_thesis_section()
    
    # 3. Raw Database
    start_date_label = df_all['timestamp'].min().strftime('%d %b %Y')
    with st.expander(f"📂 View Raw Database (Data from {start_date_label})"):
        st.dataframe(df_all.style.format("{:.2f}", subset=['spot_price', 'm1_straddle', 'm1_iv', 'm2_iv', 'm3_iv', 'skew_index', 'india_vix']))

if __name__ == "__main__":
    main()
