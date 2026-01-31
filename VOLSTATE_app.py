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
    
    /* Executive Summary Box */
    .exec-box {
        background-color: #1e252e;
        border: 2px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cis-score { font-size: 48px; font-weight: 900; color: #fff; }
    .cis-label { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .status-badge { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 20px; text-transform: uppercase; color: #000;}

    /* Regime Box (Demoted) */
    .regime-box { 
        text-align: center; 
        padding: 15px; 
        border-radius: 12px; 
        margin-bottom: 25px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .regime-label { font-size: 24px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
    .regime-sub { font-size: 12px; opacity: 0.8; margin-top: 5px; font-family: monospace; }
    
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
    
    /* Utility classes for Theta Cycle Highlighting */
    .cycle-highlight { color: #ddd; font-weight: bold; }
    .cycle-dim { color: #555; text-decoration: line-through; font-size: 0.9em;}
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

# --- 5️⃣ CIS ENGINE (NEW) ---

def compute_cis_score(rpv, drift, stress_accel, std_pct, m1_iv, m2_iv):
    # Inputs
    C = rpv['COMPRESSION']
    T = rpv['TRANSITION']
    S = rpv['STRESS']
    
    # 1. Theta Component: Positive if decay (std_pct negative)
    # Target: 0.25% daily decay
    theta_raw = -std_pct
    theta_hat = np.clip(theta_raw / 0.25, -1, 1)
    
    # 2. Insulation Component: M2 - M1
    # Target: 1.0 vol point buffer
    k_raw = m2_iv - m1_iv
    k_hat = np.clip(k_raw / 1.0, -1, 1)
    
    # 3. Drift Component
    d_raw = drift['STRESS']
    d_hat = np.clip(d_raw / 0.10, 0, 1)
    
    # 4. Accel Component
    a_val = 1.0 if stress_accel else 0.0
    
    # FORMULA
    # CIS = +0.40(C+T) + 0.20(Theta) + 0.15(K) - 0.50(S) - 0.30(Drift) - 0.20(Accel)
    cis_val = (0.40 * (C + T)) + (0.20 * theta_hat) + (0.15 * k_hat) - (0.50 * S) - (0.30 * d_hat) - (0.20 * a_val)
    
    return np.clip(cis_val, -1, 1)

def get_cis_status(score):
    if score > 0.35: return "SAFE", "#28a745"
    if score > 0.15: return "CAUTION", "#ffc107"
    if score > 0.0: return "DEFENSIVE", "#fd7e14"
    if score > -0.25: return "EXIT BIAS", "#dc3545"
    return "IMMEDIATE EXIT", "#dc3545"

# --- ENGINE WRAPPERS ---

def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 5: return None, None, df_c.iloc[-1]

    # Current Snapshot
    curr, prev, prev2 = df_c.iloc[-1], df_c.iloc[-2], df_c.iloc[-3]
    rpv, lhs = compute_rpv(curr, prev, prev2)
    dominant = max(rpv, key=rpv.get)
    
    colors = {"COMPRESSION": "#28a745", "TRANSITION": "#ffc107", "EXPANSION": "#fd7e14", "STRESS": "#dc3545"}
    color = colors.get(dominant, "#888")
    
    entropy = regime_entropy(rpv)
    if rpv[dominant] > 0.55 and entropy < 1.1: confidence = "HIGH"
    elif rpv[dominant] > 0.4: confidence = "MEDIUM"
    else: confidence = "LOW (UNSTABLE)"

    # History for Dynamics
    rpv_df_hist = compute_rpv_series(df_c.tail(15))
    drift = compute_rpv_drift(rpv_df_hist)
    pre_stress, ps_details = detect_pre_stress(rpv_df_hist)
    
    # CIS Calculation
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    m1 = curr['m1_iv']
    m2 = curr.get('m2_iv', m1) # Fallback if missing
    
    cis_score = compute_cis_score(rpv, drift, pre_stress, std_pct, m1, m2)
    cis_label, cis_color = get_cis_status(cis_score)

    risk_posture = derive_risk_posture(rpv)
    drivers = [k for k, v in lhs.items() if v[dominant] > 0.6] # Simplified
    counterforces = [k for k, v in lhs.items() if v[dominant] < 0.3]
    
    dte = curr.get('m1_dte', 30)
    is_roll, is_late = dte >= 28, dte <= 7
    
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    slope = curr['m3_iv'] - curr['m1_iv'] # Kept for display, but CIS uses M2
    skew_chg = curr['skew_index'] - prev['skew_index']
    m2_m1_spread = m2 - m1
    
    # RENAMED SIGNALS FOR UI
    signals = {
        't1': (iv_chg > 0.2, "RISING" if iv_chg > 0.2 else "STABLE", f"{iv_chg:+.2f}%"), # Front Stress
        't2': (std_pct > -0.1, "STALLED" if std_pct > -0.1 else "DECAYING", f"{std_pct:+.2f}%"), # Theta Eff
        't4': (m2_m1_spread < 0, "INVERTED" if m2_m1_spread < 0 else "NORMAL", f"{m2_m1_spread:.2f}"), # Carry Insulation
        't5': (skew_chg > 0.3, "RISING" if skew_chg > 0.3 else "FLAT", f"{skew_chg:+.2f}"), # Crash Hedge
    }

    # Theta Cycle Flags
    entry_allowed = (cis_score >= 0.15) and (rpv['COMPRESSION'] + rpv['TRANSITION'] >= 0.55)
    harvest_warning = (drift['STRESS'] > 0.05) or (rpv['STRESS'] > 0.15)
    exit_trigger = (cis_score < 0) or (rpv['STRESS'] >= 0.25) or pre_stress

    cycle_status = {
        "entry": ("✅ SAFE", "#28a745") if entry_allowed else ("🛑 NO ENTRY", "#dc3545"),
        "harvest": ("⚠️ CAUTION", "#ffc107") if harvest_warning else ("✅ STABLE", "#28a745"),
        "exit": ("🚨 EXIT NOW", "#dc3545") if exit_trigger else ("HOLD", "#888")
    }

    context = {
        'regime': dominant, 'color': color, 'confidence': confidence,
        'rpv': rpv, 'risk': risk_posture, 'drivers': drivers, 'counterforces': counterforces,
        'is_roll': is_roll, 'is_late': is_late, 'dte': dte,
        'cycle': cycle_status,
        'cis': {'score': cis_score, 'label': cis_label, 'color': cis_color},
        'entry_bool': entry_allowed,
        'harvest_bool': harvest_warning,
        'exit_bool': exit_trigger
    }
    return signals, context, curr

def calculate_historical_regime(df):
    df_sorted = df.sort_values('timestamp', ascending=True).copy()
    history = []
    if len(df_sorted) < 5: return pd.DataFrame()
    
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

# --- INFO SECTIONS ---

def render_dashboard_key():
    with st.expander("ℹ️ **Dashboard Key & Glossary (Click to Expand)**", expanded=False):
        st.markdown("""
        ### 📊 **1. Carry Integrity Score (CIS)**
        * **The Gauge:** A single score (-1.0 to +1.0) measuring the structural safety of a short gamma position.
        * **Green (> +0.35):** Full Carry Mode.
        * **Yellow (+0.15 to +0.35):** Reduced Size.
        * **Red (< 0):** Hard Exit Bias.
        
        ### 🔄 **2. Optimal Theta Cycle Control**
        * **Phase 1: Entry:** Gates your trade initiation. Only enter if "SAFE" (Green).
        * **Phase 2: Harvest:** Monitors your active position. "STABLE" means let theta decay work.
        * **Phase 3: Exit Trigger:** The emergency brake. If "EXIT NOW" appears, close the trade immediately.

        ### 📉 **3. Analytics Charts (Renamed)**
        
        **1. Front Stress (M1):** (Formerly IV Momentum). Measures immediate panic in the nearest expiry.
        **2. Theta Efficiency:** (Formerly Straddle Decay). Is the straddle losing value fast enough to justify the risk?
        **3. Carry Insulation (M2-M1):** (Formerly Term Structure). The buffer between your trade (M2) and the storm (M1).
        **4. Crash Hedging Pressure:** (Formerly Skew). Are institutions buying OTM Puts aggressively?
        """, unsafe_allow_html=True)

def render_volstate_methodology():
    with st.expander("📘 **VOLSTATE Methodology**", expanded=False):
        st.markdown("""
        # 📘 The VOLSTATE Methodology

        This engine uses Fuzzy Logic. Unlike binary logic (True/False), Fuzzy Logic deals with "degrees of truth."

        - Binary: "Is IV rising?" -> Yes (1) / No (0).
        - Fuzzy: "Is IV rising?" -> Somewhat (0.4), Yes (0.8), Extremely (1.0).

        We use 6 "Likelihood Functions" to convert raw market data into these probabilistic scores.

        ## 1. The Likelihood Functions (The "Sensors")

        ### Sensor 1: ATM IV Momentum (iv_likelihood)
        **Question:** Is the cost of options (Implied Volatility) rising?
        - **Input:** iv_chg (Today's IV - Yesterday's IV).
        - **The Logic:**
            - **Compression** hates rising IV. If IV goes up, Compression score drops to 0.
            - **Stress** loves rising IV. If IV spikes > 0.8, Stress score hits 1.0.
            - **Expansion** tolerates rising IV (it needs movement).

        ### Sensor 2: Straddle Decay (straddle_likelihood)
        **Question:** Is the ATM Straddle losing value?
        - **Input:** std_pct (% Change in Straddle Price).
        - **The Logic:**
            - **Compression:** Requires the straddle to lose value (Theta decay > Vega/Gamma). If price drops significantly (< -0.2%), score is 1.0.
            - **Stress:** Occurs when the straddle gains value rapidly (Vega explosion).

        ### Sensor 3: Back-Month Spread (back_month_likelihood)
        **Question:** Are traders bidding up the next month's volatility relative to this month?
        - **Input:** bm_spread (Change in Far IV - Change in Near IV).
        - **The Logic:**
            - If Far IV is rising faster than Near IV, traders fear the future (**Transition/Expansion**).
            - If Near IV is rising faster than Far IV, traders fear the now (**Stress/Inversion**).
            - If Far IV is falling faster, the curve is steepening (**Compression**).

        ### Sensor 4: Term Structure (term_likelihood)
        **Question:** Is the curve normal (Contango) or inverted (Backwardation)?
        - **Input:** slope (Far IV - Near IV).
        - **The Logic:**
            - **Normal Market (Contango):** Far months are expensive (uncertainty). Slope > 0. This is Compression.
            - **Panic Market (Inversion):** Near months are expensive (immediate fear). Slope < 0. This is Stress.

        ### Sensor 5: Skew Index (skew_likelihood)
        **Question:** How expensive are Puts vs. Calls?
        - **Input:** skew_chg (Change in Skew).
        - **The Logic:**
            - **Rising Skew:** People are buying crash protection (Puts). This signals Stress or Expansion.
            - **Falling Skew:** People are selling Puts or buying Calls. This signals Compression (Complacency).
            - **Why weight 1.4?** Skew is the "smart money" indicator. It is harder to fake than Spot Price.

        ### Sensor 6: Spot-Vol Disconnect (disconnect_likelihood)
        **Question:** Is the market broken?
        - **Input:** is_disconnect (Boolean: Price UP + Vol UP).
        - **The Logic:**
            - **Normal Physics:** Price UP -> Vol DOWN.
            - **Broken Physics:** Price UP -> Vol UP.
            - This "Disconnect" usually happens right before a crash (euphoria) or during a squeeze. It strongly weights Stress (0.9) and Expansion (0.7).

        ## 2. The Aggregation Engine (compute_rpv)
        Once the sensors give us scores, we combine them.

        **Step A: Weighted Sum**
        We don't treat all sensors equally.
        - High Confidence: Skew (1.4), IV (1.2), Straddle (1.2).
        - Low Confidence: Disconnect (0.8) - it's noisy.

        **Step B: Normalization**
        We turn raw scores into a percentage (Probability).

        ## 3. The Entropy Filter (The "BS Detector")
        This helps us know if the RPV is trustworthy.
        - If RPV is {0.9, 0.0, 0.1, 0.0}, the model is certain. (**Low Entropy**)
        - If RPV is {0.25, 0.25, 0.25, 0.25}, the model is confused. (**High Entropy**)

        **Application in Dashboard:**
        We only label confidence as "HIGH" if:
        1. The winner has > 55% probability.
        2. AND Entropy is low (< 1.1).
        """
        )

def render_theta_cycle_philosophy():
    with st.expander("🔄 **Optimal Theta Cycle Philosophy**", expanded=False):
        st.markdown("""
        ### 🎯 The Core Philosophy: "Permission, Not Prediction"
        You are not a direction trader. You are a **Structured VRP Harvester**. 
        Your edge comes from selling time and variance, surviving transitions, and not overstaying when the regime turns hostile.
        
        **VOLSTATE's Job:** 1. Gate entries. 2. Modulate size. 3. Force early exits when structure breaks.

        ---

        ### 🟢 Phase 1: Entry Window (≈ 45 DTE)
        **Action:** Initiate VRP Harvest.
        **The "Entry Gate" Criteria (Must Pass ALL):**
        1.  **Structure:** `Compression + Transition ≥ 55%`
        2.  **Safety:** `Stress ≤ 15%`
        3.  **Clarity:** Entropy is NOT High (Confidence ≠ LOW).
        4.  **Momentum:** `Stress Drift ≤ +0.05` (over last 3 bars).

        *If these pass → Enter normally.*
        *If borderline → Enter smaller.*
        *If fail → Skip the cycle entirely.*

        ---

        ### 🟡 Phase 2: Harvest Window (45 → 30 DTE)
        **Action:** Let Theta work. Monitor weekly.
        **The "Risk Thermostat":**
        * **Stress < 15%:** ✅ Full Carry Allowed.
        * **Stress 15-20%:** ⚠️ Freeze new risk. Stop adding.
        * **Stress > 20%:** 🛑 Defensive actions required.
        
        **Watch For:**
        * **Stress Slope:** If rising steadily, it's a warning.
        * **Skew Spike:** If Skew Likelihood > 0.6 but price is calm, institutions are hedging ahead of you. Take the hint.

        ---

        ### 🔴 Phase 3: Exit Window (30 → 21 DTE)
        **Action:** Hard exit at 21 DTE.
        **The "Early Exit Override" (Fire Exit if ANY occur):**
        1.  **Stress ≥ 25%** (The "Hard Deck").
        2.  **Stress Drift > +0.08** (Acceleration).
        3.  **Expansion + Stress ≥ 60%** AND Straddle Stalls.
        4.  **Entropy Spikes** (Confidence flips to LOW).

        *If any of these fire, exit immediately, even at 35 DTE. Do not wait for the hard exit.*
        """)

def render_dashboard(df_selected, signals, ctx, curr, df_all):
    # --- EXECUTIVE SUMMARY (CIS) ---
    cis = ctx['cis']
    st.markdown(f"""
    <div class="exec-box" style="border-color: {cis['color']};">
        <div>
            <div style="font-size: 14px; color: #888;">EXECUTIVE COMMAND</div>
            <div class="status-badge" style="background-color: {cis['color']};">{cis['label']}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 14px; color: #888;">CARRY INTEGRITY SCORE</div>
            <div class="cis-score">{cis['score']*100:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- REGIME BANNER (DEMOTED) ---
    p_comp = int(ctx['rpv']['COMPRESSION'] * 100)
    p_tran = int(ctx['rpv']['TRANSITION'] * 100)
    p_expa = int(ctx['rpv']['EXPANSION'] * 100)
    p_strs = int(ctx['rpv']['STRESS'] * 100)

    rpv_html = f"""<div class="regime-box" style="background-color: {ctx['color']}15; border-color: {ctx['color']}80;">
<div style="text-align: center; font-size: 11px; color: #888; letter-spacing: 1px; font-weight: 600;">MARKET STRUCTURE (DIAGNOSTIC)</div>
<div class="regime-label" style="color: {ctx['color']}; font-size: 20px;">{ctx['regime']} <span style='font-size: 12px; color: #aaa'>({ctx['confidence']})</span></div>

<div class="rpv-bar" style="display: flex; height: 6px; background: #222; border-radius: 4px; overflow: hidden; margin-top: 10px; width: 100%;">
    <div style="width: {ctx['rpv']['COMPRESSION']*100}%; background: #28a745;" title="Compression"></div>
    <div style="width: {ctx['rpv']['TRANSITION']*100}%; background: #ffc107;" title="Transition"></div>
    <div style="width: {ctx['rpv']['EXPANSION']*100}%; background: #fd7e14;" title="Expansion"></div>
    <div style="width: {ctx['rpv']['STRESS']*100}%; background: #dc3545;" title="Stress"></div>
</div>

<div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; font-family: monospace; font-weight: bold;">
    <div style="color: #28a745;">COMP: {p_comp}%</div>
    <div style="color: #ffc107;">TRAN: {p_tran}%</div>
    <div style="color: #fd7e14;">EXPA: {p_expa}%</div>
    <div style="color: #dc3545;">STRS: {p_strs}%</div>
</div>

</div>"""
    st.markdown(rpv_html, unsafe_allow_html=True)

    # --- OPTIMAL THETA CYCLE CONTROL UI ---
    cyc = ctx['cycle']
    
    # Reason Validation Logic strings
    p1_sub = "Edge > 55% & Stress Low" if ctx['entry_bool'] else "Edge Low / Stress High"
    p2_sub = "Stress Drift > 0.05" if ctx['harvest_bool'] else "Drift Stable"
    p3_sub = "Stress > 0.25 or Accel" if ctx['exit_bool'] else "Limits Not Breached"

    if ctx['entry_bool']:
        p1_html = "<span class='cycle-highlight'>Edge > 55% & Stress Low</span><br><span class='cycle-dim'>Edge Low / Stress High</span>"
    else:
        p1_html = "<span class='cycle-dim'>Edge > 55% & Stress Low</span><br><span class='cycle-highlight' style='color:#dc3545'>Edge Low / Stress High</span>"
    
    if ctx['harvest_bool']:
        p2_html = "<span class='cycle-highlight' style='color:#ffc107'>Stress Drift > 0.05</span><br><span class='cycle-dim'>Drift Stable</span>"
    else:
        p2_html = "<span class='cycle-dim'>Stress Drift > 0.05</span><br><span class='cycle-highlight'>Drift Stable</span>"
    
    if ctx['exit_bool']:
        p3_html = "<span class='cycle-highlight' style='color:#dc3545'>Stress > 0.25 or Accel</span><br><span class='cycle-dim'>Limits Not Breached</span>"
    else:
        p3_html = "<span class='cycle-dim'>Stress > 0.25 or Accel</span><br><span class='cycle-highlight'>Limits Not Breached</span>"

    st.markdown(f"""
    <div style="margin-bottom: 25px; padding: 15px; background: #161b22; border: 1px solid #444; border-radius: 8px;">
        <div style="font-size: 14px; font-weight: bold; color: #fff; margin-bottom: 10px; text-transform: uppercase;">Optimal Theta Cycle Control</div>
        <div style="display: flex; justify-content: space-between; gap: 15px;">
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">PHASE 1: ENTRY</div>
                <div style="font-size: 16px; font-weight: bold; color: {cyc['entry'][1]}; margin-bottom: 5px;">{cyc['entry'][0]}</div>
                <div style="font-size: 10px; color: #aaa; line-height: 1.4;">{p1_html}</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">PHASE 2: HARVEST</div>
                <div style="font-size: 16px; font-weight: bold; color: {cyc['harvest'][1]}; margin-bottom: 5px;">{cyc['harvest'][0]}</div>
                <div style="font-size: 10px; color: #aaa; line-height: 1.4;">{p2_html}</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">PHASE 3: EXIT TRIGGER</div>
                <div style="font-size: 16px; font-weight: bold; color: {cyc['exit'][1]}; margin-bottom: 5px;">{cyc['exit'][0]}</div>
                <div style="font-size: 10px; color: #aaa; line-height: 1.4;">{p3_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 4 COLUMN SIGNALS (RENAMED) ---
    c1, c2, c3, c4 = st.columns(4)
    with c1: s = signals['t1']; render_tile("FRONT STRESS (M1)", s[0], s[1], s[2])
    with c2: s = signals['t2']; render_tile("THETA EFFICIENCY", s[0], s[1], s[2])
    with c3: s = signals['t4']; render_tile("CARRY INSULATION (M2-M1)", s[0], s[1], s[2], is_stress=True)
    with c4: s = signals['t5']; render_tile("CRASH HEDGING PRESSURE", s[0], s[1], s[2])

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MINI DIAGNOSTICS ---
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
    
    # 2. Methodology Tabs
    render_volstate_methodology()
    render_theta_cycle_philosophy()
    
    # 3. Raw Database
    start_date_label = df_all['timestamp'].min().strftime('%d %b %Y')
    with st.expander(f"📂 View Raw Database (Data from {start_date_label})"):
        st.dataframe(df_all.style.format("{:.2f}", subset=['spot_price', 'm1_straddle', 'm1_iv', 'm2_iv', 'm3_iv', 'skew_index', 'india_vix']))

if __name__ == "__main__":
    main()
