import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import textwrap

# --- IMPORT DOCUMENTATION MODULE ---
try:
    from VOLSTATE_docs import render_documentation_tab
except ImportError:
    def render_documentation_tab():
        st.error("VOLSTATE_docs.py not found. Please ensure the file exists in the same directory.")

# --- CONFIG ---
DB_NAME = "market_data.db"

st.set_page_config(
    page_title="VOLSTATE System", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { font-family: 'Segoe UI', sans-serif; background-color: #0e1117; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #1e252e;
        border-radius: 5px; color: #fff; font-weight: 600; font-size: 16px; flex: 1;
    }
    .stTabs [aria-selected="true"] { background-color: #ffc107; color: #000; }

    /* Pills & Badges */
    .pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #fff; display: inline-block; }
    .pill-yellow { background-color: #ffc107; color: #000; }
    .pill-red { background-color: #dc3545; }
    .pill-orange { background-color: #fd7e14; }
    .pill-gray { background-color: #444; opacity: 0.5; }
    
    /* Exec Box */
    .exec-box { 
        background-color: #1e252e; 
        border: 2px solid #444; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 20px; 
        display: flex; 
        justify-content: space-around; 
        align-items: center; 
    }
    .cis-score { font-size: 42px; font-weight: 900; color: #fff; line-height: 1; }
    .status-badge { padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 16px; text-transform: uppercase; color: #000; display: inline-block; }
    .delta-arrow { font-size: 24px; font-weight: bold; }
    .cis-context { font-size: 12px; color: #888; margin-top: 5px; font-style: italic; }

    /* Permission Meter - CSS kept for compatibility but element removed from DOM */
    .perm-meter {
        padding: 10px 15px; 
        border-radius: 8px; 
        background:#161b22; 
        min-width: 160px;
        text-align: center;
    }

    /* Divergence Warning Banner */
    .div-warning {
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        font-size: 16px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Regime Box */
    .regime-box { text-align: center; padding: 15px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); background-color: rgba(255, 255, 255, 0.05); }
    .regime-label { font-size: 24px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
    
    /* Tiles */
    .grid-tile { background-color: #161b22; border: 1px solid #333; border-radius: 6px; padding: 15px; height: 110px; display: flex; flex-direction: column; justify-content: space-between; }
    .tile-header { font-size: 13px; color: #888; text-transform: uppercase; font-weight: 600; }
    .tile-value { font-size: 24px; font-weight: 800; margin: 2px 0; }
    .tile-sub { font-size: 12px; font-family: monospace; color: #aaa; }
    
    /* Utility */
    .text-green { color: #28a745; } .text-amber { color: #ffc107; } .text-red { color: #dc3545; } .text-gray { color: #888; }
    .border-green { border-left: 4px solid #28a745; } .border-amber { border-left: 4px solid #ffc107; } .border-red { border-left: 4px solid #dc3545; } .border-gray { border-left: 4px solid #555; }
    
    .mini-diag { font-family: monospace; font-size: 12px; color: #666; border-top: 1px solid #333; margin-top: 20px; padding-top: 10px; display: flex; justify-content: space-around; }
    .section-header { margin-top: 40px; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #333; font-size: 20px; font-weight: bold; color: #ddd;}
    .rpv-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 10px; width: 100%; }
    .stDateInput label { display: none; }
    
    /* DYNAMICS CONSOLE (VIBRANT UPDATE) */
    .dynamics-console {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #555;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 25px;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .dynamics-title {
        font-size: 12px; color: #aaa; font-weight: bold; letter-spacing: 1.5px; margin-bottom: 8px; text-transform: uppercase;
    }
    
    /* Pre-Stress Panel */
    .ps-panel {
        flex: 1;
        min-width: 200px;
        border-right: 1px solid #444;
        padding-right: 20px;
    }
    /* Glow Effects for Status */
    .ps-status-safe { 
        color: #00e676; 
        font-size: 22px; 
        font-weight: 900; 
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.2); 
    }
    .ps-status-danger { 
        color: #ff1744; 
        font-size: 22px; 
        font-weight: 900; 
        animation: pulse 1.5s infinite; 
        text-shadow: 0 0 10px rgba(255, 23, 68, 0.3);
    }
    
    .ps-metrics { font-family: monospace; font-size: 13px; color: #bbb; margin-top: 8px; }
    
    /* Drift Grid */
    .drift-grid {
        flex: 3;
        display: flex;
        justify-content: space-between;
        gap: 15px;
    }
    .drift-item {
        background: #0d1117;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px;
        flex: 1;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.3);
    }
    .drift-item:hover {
        border-color: #777;
        transform: translateY(-2px);
    }
    .drift-label { font-size: 11px; color: #999; font-weight: bold; letter-spacing: 0.5px; }
    .drift-val { font-size: 18px; font-weight: bold; margin-top: 4px; }
    
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADERS ---
def load_data(limit=300):
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM market_logs ORDER BY timestamp DESC LIMIT {limit}"
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cols = ['m1_iv', 'm3_iv', 'm1_straddle', 'spot_price', 'm2_iv', 'skew_index', 'india_vix']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df

# --- UI HELPER ---
def render_tile(label, state_bool, display_text, subtext, is_stress=False):
    if state_bool is None: c, b = "text-gray", "border-gray"
    elif state_bool: c, b = ("text-red", "border-red") if is_stress else ("text-amber", "border-amber")
    else: c, b = "text-green", "border-green"
    st.markdown(f"""<div class="grid-tile {b}"><div class="tile-header">{label}</div><div class="tile-value {c}">{display_text}</div><div class="tile-sub">{subtext}</div></div>""", unsafe_allow_html=True)

# --- ENGINE LOGIC ---
def iv_likelihood(iv_chg): return {"COMPRESSION": max(0, 1 - iv_chg/0.5), "TRANSITION": np.clip(iv_chg/0.6, 0, 1), "EXPANSION": np.clip(iv_chg/1.0, 0, 1), "STRESS": np.clip((iv_chg - 0.8)/1.2, 0, 1)}
def straddle_likelihood(std_pct): return {"COMPRESSION": 1 if std_pct < -0.2 else 0.2, "TRANSITION": np.clip((std_pct + 0.2)/0.4, 0, 1), "EXPANSION": np.clip((std_pct + 0.1)/0.6, 0, 1), "STRESS": np.clip((std_pct - 0.8)/1.0, 0, 1)}
def back_month_likelihood(bm_spread): return {"COMPRESSION": np.clip(-bm_spread / 0.3, 0, 1), "TRANSITION": np.clip(bm_spread/0.4, 0, 1), "EXPANSION": np.clip(bm_spread/0.6, 0, 1), "STRESS": np.clip(bm_spread/0.8, 0, 1)}
def term_likelihood(slope): return {"COMPRESSION": np.clip(slope/1.5, 0, 1), "TRANSITION": np.clip((1.2 - slope)/1.2, 0, 1), "EXPANSION": np.clip((0.8 - slope)/0.8, 0, 1), "STRESS": np.clip((-slope - 0.3)/0.7, 0, 1)}
def skew_likelihood(skew_chg): return {"COMPRESSION": 1 if skew_chg <= 0 else 0, "TRANSITION": np.clip((0.3 - skew_chg)/0.3, 0, 1), "EXPANSION": np.clip(skew_chg/0.6, 0, 1), "STRESS": np.clip((skew_chg - 0.6)/1.0, 0, 1)}
def disconnect_likelihood(disc): return {"COMPRESSION": 0, "TRANSITION": 0.4 if disc else 0.6, "EXPANSION": 0.7 if disc else 0.4, "STRESS": 0.9 if disc else 0.2}

REGIMES = ["COMPRESSION", "TRANSITION", "EXPANSION", "STRESS"]
WEIGHTS = {"iv": 1.2, "straddle": 1.2, "back_month": 1.0, "term": 1.0, "skew": 1.4, "disconnect": 0.8}

def compute_rpv(curr, prev, prev2):
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    bm_spread = (curr.get('m2_iv', 0) - prev.get('m2_iv', 0)) - iv_chg
    slope = curr['m3_iv'] - curr['m1_iv']
    skew_chg = curr['skew_index'] - prev['skew_index']
    disc = (abs(((curr['spot_price'] - prev['spot_price']) / prev['spot_price']) * 100) < 0.15 and iv_chg > 0.5)
    
    lhs = {
        "iv": iv_likelihood(iv_chg), "straddle": straddle_likelihood(std_pct), "back_month": back_month_likelihood(bm_spread),
        "term": term_likelihood(slope), "skew": skew_likelihood(skew_chg), "disconnect": disconnect_likelihood(disc)
    }
    scores = {r: sum(WEIGHTS[k] * lhs[k][r] for k in lhs) for r in REGIMES}
    total = sum(scores.values())
    return {r: (scores[r]/total if total > 0 else 0) for r in REGIMES}, lhs

def regime_entropy(rpv): return -sum(p * np.log(p + 1e-9) for p in rpv.values())

def derive_risk_posture(rpv):
    if rpv["STRESS"] > 0.20: h = ("MANDATORY", "#dc3545")
    elif rpv["STRESS"] > 0.15: h = ("ACCUMULATE", "#ffc107")
    else: h = ("OPTIONAL", "#666")
    return {"long_gamma": (rpv["EXPANSION"] + rpv["STRESS"] > 0.5) and (rpv["COMPRESSION"] < 0.3), "short_theta": rpv["COMPRESSION"] > 0.5, "tail_hedge_data": h, "carry_allowed": rpv["COMPRESSION"] > 0.4 and rpv["STRESS"] < 0.15}

def compute_rpv_series(df):
    rows = []
    if len(df) < 3: return pd.DataFrame()
    for i in range(2, len(df)):
        rpv, _ = compute_rpv(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
        rpv['timestamp'] = df.iloc[i]['timestamp']
        rows.append(rpv)
    return pd.DataFrame(rows)

def compute_rpv_drift(rpv_df, lookback=3):
    return {r: (rpv_df[r].iloc[-1] - rpv_df[r].iloc[-(lookback+1)]) if len(rpv_df) > lookback else 0.0 for r in REGIMES}

def detect_pre_stress(rpv_df):
    if len(rpv_df) < 4: return False, {"stress_slope": 0, "stress_accel": False, "stress_val": 0, "exp_val": 0}
    s, e = rpv_df["STRESS"].values, rpv_df["EXPANSION"].values
    slope = s[-1] - s[-3]
    accel = (s[-1] - s[-2]) > (s[-2] - s[-3])
    is_triggered = (s[-1] > 0.20 and slope > 0.08 and accel and s[-1] > e[-1] * 0.6)
    return is_triggered, {"stress_slope": slope, "stress_accel": accel, "stress_val": s[-1], "exp_val": e[-1]}

# --- CIS CONTEXT LOGIC ---
def cis_context_label(cis, rpv, drift, pre_stress, std_pct, iv_chg, term_spread):
    if rpv['STRESS'] > 0.20:
        return "Low CIS due to rising stress probability"
    if pre_stress or drift['STRESS'] > 0.08:
        return "Low CIS due to accelerating stress drift"
    if std_pct > -0.10:
        return "Low CIS due to stalled straddle decay"
    if iv_chg > 0.25:
        return "Low CIS due to front-month IV repricing"
    if term_spread < 0:
        return "Low CIS due to term structure erosion"
    return "Carry structure stable"

# --- CARRY BAND LOGIC (FIXED) ---
def carry_permission_band(cis):
    if cis > 0.35:
        return "FULL CARRY", "#28a745"
    if cis > 0.15:
        return "CONTROLLED CARRY", "#ffc107"
    if cis > -0.05:
        return "TOLERANCE ONLY", "#fd7e14"  # ORANGE
    return "NO CARRY", "#dc3545"          # RED

# --- CIS ENGINE ---
def compute_cis_score(rpv, drift, stress_accel, std_pct, m1, m2):
    return np.clip(0.40*(rpv['COMPRESSION']+rpv['TRANSITION']) + 0.20*np.clip(-std_pct/0.25, -1, 1) + 0.15*np.clip((m2-m1)/1.0, -1, 1) - 0.50*rpv['STRESS'] - 0.30*np.clip(drift['STRESS']/0.10, 0, 1) - 0.20*(1.0 if stress_accel else 0.0), -1, 1)

# REMOVED OLD get_cis_status TO AVOID CONFUSION

# --- CPS ENGINE (UPDATED: DUAL LAYER) ---
def compute_cps_score(rpv, std_pct, skew_accel_bool, m1, m2, m3):
    struct = 0.40 * (rpv['EXPANSION'] + rpv['STRESS'])
    gamma_hat = 0.25 * np.clip(std_pct / 0.30, -1, 1)
    term_hat = 0.15 * np.clip(-(m3 - m1)/1.0, 0, 1)
    lag_convexity = 0.10 * np.clip((m2 - m1) / 0.8, -1, 1)
    accel_hat = 0.10 * (1.0 if skew_accel_bool else 0.0)
    cps = struct + gamma_hat + term_hat + lag_convexity + accel_hat
    return np.clip(cps, -1, 0.8)

def get_cps_status(score):
    if score > 0.40: return "CONVEXITY PERMITTED", "#28a745"
    if score > 0.15: return "SELECTIVE", "#ffc107"
    if score > -0.15: return "NEUTRAL", "#888"
    return "EXPENSIVE", "#dc3545"

# --- METRICS: DECAY & DIVERGENCE ---
def permission_decay_meter(cis_series):
    if len(cis_series) < 4: return 0.0
    delta = cis_series[-1] - cis_series[-4]
    decay = np.clip(-delta / 0.15, 0, 1) 
    return decay

def get_decay_status(decay):
    if decay < 0.25: return "#28a745", "STABLE"
    if decay < 0.5: return "#ffc107", "DECAYING"
    if decay < 0.75: return "#fd7e14", "DANGEROUS"
    return "#dc3545", "EXIT ZONE"

def check_cis_divergence(cis, cis_prev, std_pct, vrp, rpv, rpv_prev):
    if std_pct < -0.15 and cis < cis_prev: return "⚠️ PERMISSION DECAY (Theta Trap)", "#ffc107"
    stress_delta = abs(rpv['STRESS'] - rpv_prev['STRESS'])
    if stress_delta < 0.03 and (cis - cis_prev) < -0.1: return "⚠️ STRUCTURAL ROT (Hidden Risk)", "#fd7e14"
    if vrp > 0 and cis < 0: return "🚨 EDGE MIRAGE (Convexity Risk)", "#dc3545"
    if cis > 0.25 and (rpv['EXPANSION'] + rpv['STRESS']) > 0.6: return "🚨 SYSTEM CONFLICT (Data Error)", "#dc3545"
    return None, None

# --- PLOT CIS/CPS TREND ---
def plot_cis_cps_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cis'], mode='lines+markers', name='CIS (Carry)', line=dict(color='#00d1b2', width=2)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cps'], mode='lines+markers', name='CPS (Convexity)', line=dict(color='#e67e22', width=2, dash='dot')))
    fig.add_hline(y=0, line_dash='dash', line_color='#666')
    fig.update_layout(title="<b>CIS vs CPS — Permission Regimes</b>", template="plotly_dark", height=280, margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(range=[-1, 1], title="Permission"), legend=dict(orientation="h", y=1.1))
    return fig

def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 5: return None, None, df_c.iloc[-1]
    
    # MODIFICATION: Rolling 5-Day RV for VRP
    df_c['log_ret'] = np.log(df_c['spot_price'] / df_c['spot_price'].shift(1))
    df_c['rv_5d'] = df_c['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
    
    # 1. HISTORICAL CALCULATION (CIS & CPS)
    history_data = []
    hist_window = df_c.tail(60)
    
    for i in range(3, len(hist_window)): 
        _c, _p, _p2, _p3 = hist_window.iloc[i], hist_window.iloc[i-1], hist_window.iloc[i-2], hist_window.iloc[i-3]
        _rpv, _ = compute_rpv(_c, _p, _p2)
        _std_pct = ((_c['m1_straddle'] - _p['m1_straddle']) / _p['m1_straddle']) * 100
        _cis = compute_cis_score(_rpv, {'STRESS':0}, False, _std_pct, _c['m1_iv'], _c.get('m2_iv', _c['m1_iv']))
        _skew_accel = (_c['skew_index'] - _p['skew_index']) > (_p['skew_index'] - _p2['skew_index'])
        _cps = compute_cps_score(_rpv, _std_pct, _skew_accel, _c['m1_iv'], _c.get('m2_iv', _c['m1_iv']), _c['m3_iv'])
        history_data.append({'timestamp': _c['timestamp'], 'cis': _cis, 'cps': _cps, 'rpv': _rpv})
        
    df_hist = pd.DataFrame(history_data)

    # 2. CURRENT CALCULATIONS
    curr, prev, prev2, prev3 = df_c.iloc[-1], df_c.iloc[-2], df_c.iloc[-3], df_c.iloc[-4]
    rpv, lhs = compute_rpv(curr, prev, prev2)
    dom = max(rpv, key=rpv.get)
    colors = {"COMPRESSION": "#28a745", "TRANSITION": "#ffc107", "EXPANSION": "#fd7e14", "STRESS": "#dc3545"}
    
    rpv_hist_short = compute_rpv_series(df_c.tail(15))
    drift = compute_rpv_drift(rpv_hist_short)
    pre_stress, ps_det = detect_pre_stress(rpv_hist_short)
    
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    skew_accel_bool = (curr['skew_index'] - prev['skew_index']) > (prev['skew_index'] - prev2['skew_index'])
    
    cis = compute_cis_score(rpv, drift, pre_stress, std_pct, curr['m1_iv'], curr.get('m2_iv', curr['m1_iv']))
    cps = compute_cps_score(rpv, std_pct, skew_accel_bool, curr['m1_iv'], curr.get('m2_iv', curr['m1_iv']), curr['m3_iv'])
    
    if not df_hist.empty and cis != df_hist.iloc[-1]['cis']:
         new_row = pd.DataFrame([{'timestamp': curr['timestamp'], 'cis': cis, 'cps': cps, 'rpv': rpv}])
         df_hist = pd.concat([df_hist, new_row], ignore_index=True)

    # 3. METRICS
    cis_vals = df_hist['cis'].values
    decay_val = permission_decay_meter(cis_vals)
    decay_color, decay_label = get_decay_status(decay_val)
    
    prev_cis_val = cis_vals[-2] if len(cis_vals) > 1 else cis
    prev_rpv = df_hist.iloc[-2]['rpv'] if len(df_hist) > 1 else rpv
    
    vrp_proxy = curr['m1_iv'] - curr.get('rv_5d', 0)
    div_msg, div_col = check_cis_divergence(cis, prev_cis_val, std_pct, vrp_proxy, rpv, prev_rpv)

    cis_delta = cis - prev_cis_val
    
    # --- UPDATED DTE CALCULATION ---
    try:
        # Calculate DTE: Expiry Date (from m1_month) - Current Data Date (from timestamp)
        expiry_date = pd.to_datetime(curr['m1_month'], format='%d-%b-%Y')
        current_data_date = curr['timestamp']
        dte = (expiry_date - current_data_date).days
        if dte < 0: dte = 0 # Safety floor
    except:
        dte = 30 # Fallback default

    signals = {
        't1': (curr['m1_iv'] - prev['m1_iv'] > 0.2, "RISING" if curr['m1_iv'] - prev['m1_iv'] > 0.2 else "STABLE", f"{curr['m1_iv'] - prev['m1_iv']:+.2f}%"), 
        't2': (std_pct > -0.1, "STALLED" if std_pct > -0.1 else "DECAYING", f"{std_pct:+.2f}%"), 
        't4': (curr.get('m2_iv',0) - curr['m1_iv'] < 0, "INVERTED" if curr.get('m2_iv',0) - curr['m1_iv'] < 0 else "NORMAL", f"{curr.get('m2_iv',0) - curr['m1_iv']:.2f}"), 
        't5': (curr['skew_index'] - prev['skew_index'] > 0.3, "RISING" if curr['skew_index'] - prev['skew_index'] > 0.3 else "FLAT", f"{curr['skew_index'] - prev['skew_index']:+.2f}"), 
    }
    
    # Calculate vars for Context Logic
    iv_chg_val = curr['m1_iv'] - prev['m1_iv']
    term_spread_val = curr.get('m2_iv', curr['m1_iv']) - curr['m1_iv']

    # New Context Logic Call
    context_label = cis_context_label(cis, rpv, drift, pre_stress, std_pct, iv_chg_val, term_spread_val)
    
    # *** FIX: Use correct function and capture color ***
    band_label, band_color = carry_permission_band(cis)

    ctx = {
        'regime': dom, 'color': colors.get(dom, "#888"), 'confidence': "HIGH" if rpv[dom] > 0.55 else "MEDIUM",
        'rpv': rpv, 'risk': derive_risk_posture(rpv), 'drivers': [k for k,v in lhs.items() if v[dom]>0.6], 'counterforces': [k for k,v in lhs.items() if v[dom]<0.3],
        'is_roll': dte >= 28, 'is_late': dte <= 7, 'dte': dte, 
        'cis': {'score': cis, 'label': band_label, 'color': band_color, 'delta': cis_delta, 'context': context_label},
        'cps': {'score': cps, 'label': get_cps_status(cps)[0], 'color': get_cps_status(cps)[1]},
        'decay': {'val': decay_val, 'label': decay_label, 'color': decay_color},
        'divergence': {'msg': div_msg, 'color': div_col},
        'history': df_hist,
        'drift': drift,
        'pre_stress': pre_stress,
        'ps_det': ps_det
    }
    return signals, ctx, curr

# --- HELPER: FULL RPV HISTORY ---
def get_full_rpv_history(df):
    rows = []
    if len(df) < 3: return pd.DataFrame()
    for i in range(2, len(df)):
        rpv, _ = compute_rpv(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
        rpv['timestamp'] = df.iloc[i]['timestamp']
        rows.append(rpv)
    return pd.DataFrame(rows)

# --- HELPER: CALCULATE HISTORICAL REGIME ---
def calculate_historical_regime(df):
    history = []
    if len(df) < 5: return pd.DataFrame()
    for i in range(2, len(df)):
        rpv, _ = compute_rpv(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
        dom = max(rpv, key=rpv.get)
        history.append({'timestamp': df.iloc[i]['timestamp'], 'regime': dom, 'val': 1})
    return pd.DataFrame(history)

# --- DASHBOARD RENDERER ---
def render_dashboard(df_selected, signals, ctx, curr, df_all):
    cis = ctx['cis']
    cps = ctx['cps']
    decay = ctx['decay']
    div = ctx['divergence']
    drift = ctx['drift']
    pre_stress = ctx['pre_stress']
    
    arrow = "<span class='delta-arrow' style='color: #28a745;'>↑</span>" if cis['delta'] > 0.001 else "<span class='delta-arrow' style='color: #dc3545;'>↓</span>" if cis['delta'] < -0.001 else ""
    
    # 1. Executive Command (UPDATED: REMOVED DECAY METER)
    st.markdown(f"""<div class="exec-box" style="border-color: {cis['color']};">
    <div style="flex: 1;">
        <div style="font-size: 11px; color: #888; font-weight:bold;">CARRY PERMISSION</div>
        <div class="status-badge" style="background-color: {cis['color']};">{cis['label']}</div>
        <div class="cis-score">{cis['score']*100:.0f}% {arrow}</div>
        <div class="cis-context">{cis['context']}</div>
    </div>
    <div style="flex: 1; text-align: right; padding-left: 20px;">
        <div style="font-size: 11px; color: #888; font-weight:bold;">CONVEXITY PERMISSION</div>
        <div class="status-badge" style="background-color: {cps['color']};">{cps['label']}</div>
        <div class="cis-score" style="justify-content: flex-end; color: #e67e22;">{cps['score']*100:.0f}%</div>
    </div>
</div>""", unsafe_allow_html=True)

    # 2. Market Structure
    p_comp = int(ctx['rpv']['COMPRESSION'] * 100)
    p_tran = int(ctx['rpv']['TRANSITION'] * 100)
    p_expa = int(ctx['rpv']['EXPANSION'] * 100)
    p_strs = int(ctx['rpv']['STRESS'] * 100)

    st.markdown(f"""<div class="regime-box" style="background-color: {ctx['color']}15; border-color: {ctx['color']}80;">
    <div style="text-align: center; font-size: 11px; color: #888; letter-spacing: 1px;">MARKET STRUCTURE</div>
    <div class="regime-label" style="color: {ctx['color']};">{ctx['regime']} <span style='font-size: 12px; color: #aaa'>({ctx['confidence']})</span></div>
    <div class="rpv-bar">
        <div style="width: {ctx['rpv']['COMPRESSION']*100}%; background: #28a745;"></div>
        <div style="width: {ctx['rpv']['TRANSITION']*100}%; background: #ffc107;"></div>
        <div style="width: {ctx['rpv']['EXPANSION']*100}%; background: #fd7e14;"></div>
        <div style="width: {ctx['rpv']['STRESS']*100}%; background: #dc3545;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; font-family: monospace; font-weight: bold;">
        <div style="color: #28a745;">COMP: {p_comp}%</div>
        <div style="color: #ffc107;">TRAN: {p_tran}%</div>
        <div style="color: #fd7e14;">EXPA: {p_expa}%</div>
        <div style="color: #dc3545;">STRS: {p_strs}%</div>
    </div>
</div>""", unsafe_allow_html=True)

    # 4. Divergence Warning
    if div['msg']:
        st.markdown(f"""<div class="div-warning" style="background-color: {div['color']}20; border-color: {div['color']}; color: {div['color']};">{div['msg']}</div>""", unsafe_allow_html=True)

    # 6. Signals
    c1, c2, c3, c4 = st.columns(4)
    with c1: s=signals['t1']; render_tile("FRONT STRESS (M1)", s[0], s[1], s[2])
    with c2: s=signals['t2']; render_tile("THETA EFFICIENCY", s[0], s[1], s[2])
    with c3: s=signals['t4']; render_tile("CARRY INSULATION", s[0], s[1], s[2], True)
    with c4: s=signals['t5']; render_tile("HEDGING PRESSURE", s[0], s[1], s[2])
    st.markdown(f"""<div class="mini-diag"><span>SPOT: {curr['spot_price']:.0f}</span><span>ATM IV: {curr['m1_iv']:.2f}%</span><span>STRADDLE: {curr['m1_straddle']:.0f}</span><span>DTE: {ctx['dte']}</span></div>""", unsafe_allow_html=True)

    # --- REGIME DYNAMICS (NEW SECTION) ---
    st.markdown('<div class="section-header">🔍 Regime Dynamics & Stability</div>', unsafe_allow_html=True)
    
    # Pre-Stress Status
    ps_status = "<span class='ps-status-danger'>⚠️ TRIGGERED</span>" if pre_stress else "<span class='ps-status-safe'>✅ SAFE</span>"
    ps_msg = "WARNING: TAIL RISK > 60% OF EXPANSION" if pre_stress else "System Stable. No immediate crash precursors."
    
    # Drift formatting helper
    def fmt_drift(val):
        sym = "↑" if val > 0 else "↓" if val < 0 else "−"
        c = "#28a745" if val > 0.05 else "#dc3545" if val < -0.05 else "#888"
        if val > 0.1: c = "#00e676" # strong positive
        return f"<span style='color:{c}; font-weight:bold;'>{sym} {abs(val):.2f}</span>"

    # FLUSH LEFT HTML BLOCK TO FIX INDENTATION BUG
    st.markdown(textwrap.dedent(f"""
    <div class="dynamics-console">
        <div class="ps-panel">
            <div class="dynamics-title">PRE-STRESS DETECTOR</div>
            <div class="ps-status">{ps_status}</div>
            <div class="ps-metrics">
                <div>Slope: {ctx['ps_det']['stress_slope']:.2f}</div>
                <div>Accel: {ctx['ps_det']['stress_accel']}</div>
                <div style="margin-top:4px; font-style:italic;">{ps_msg}</div>
            </div>
        </div>
        <div class="drift-grid">
            <div class="drift-item"><div class="drift-label">COMP DRIFT</div><div class="drift-val">{fmt_drift(drift['COMPRESSION'])}</div></div>
            <div class="drift-item"><div class="drift-label">TRAN DRIFT</div><div class="drift-val">{fmt_drift(drift['TRANSITION'])}</div></div>
            <div class="drift-item"><div class="drift-label">EXPA DRIFT</div><div class="drift-val">{fmt_drift(drift['EXPANSION'])}</div></div>
            <div class="drift-item"><div class="drift-label">STRS DRIFT</div><div class="drift-val">{fmt_drift(drift['STRESS'])}</div></div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # --- ANALYTICS ---
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)
    df_chart = df_selected.sort_values('timestamp').tail(60)

    # 1. Spot vs Straddle
    fig_spot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['spot_price'], line=dict(color='#3498db', width=2), name="Spot"), secondary_y=False)
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['m1_straddle'], line=dict(color='#e74c3c', width=2, dash='dot'), name="Straddle"), secondary_y=True)
    fig_spot.update_layout(title="<b>Nifty Spot vs ATM Straddle Price Trend</b>", template="plotly_dark", height=350, margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_spot, width="stretch")

    # 2. CIS vs CPS TREND
    if not ctx['history'].empty:
        fig_trend = plot_cis_cps_trend(ctx['history'])
        st.plotly_chart(fig_trend, width="stretch")

    # 3. Regime Probabilities
    rpv_full_hist = get_full_rpv_history(df_selected.tail(60))
    if not rpv_full_hist.empty:
        fig_rpv = go.Figure()
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['STRESS'], mode='lines', stackgroup='one', name='STRESS', line=dict(color='#dc3545', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['EXPANSION'], mode='lines', stackgroup='one', name='EXPANSION', line=dict(color='#fd7e14', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['TRANSITION'], mode='lines', stackgroup='one', name='TRANSITION', line=dict(color='#ffc107', width=0)))
        fig_rpv.add_trace(go.Scatter(x=rpv_full_hist['timestamp'], y=rpv_full_hist['COMPRESSION'], mode='lines', stackgroup='one', name='COMPRESSION', line=dict(color='#28a745', width=0)))
        fig_rpv.update_layout(title="<b>Regime Probabilities Over Time (2 Months)</b>", template="plotly_dark", height=300, margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(range=[0, 1]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_rpv, width="stretch")

    # 4. Metric Charts
    df_hist = df_chart.copy()
    # MODIFICATION: M2 - M1
    df_hist['slope'] = df_hist['m2_iv'] - df_hist['m1_iv']
    df_hist['slope_col'] = np.where(df_hist['slope'] >= 0, '#00cc00', '#ff0000')
    df_hist['std_pct'] = df_hist['m1_straddle'].pct_change() * 100
    df_hist['std_col'] = np.where(df_hist['std_pct'] <= 0, '#00cc00', '#ff0000') 
    df_hist['log_ret'] = np.log(df_hist['spot_price'] / df_hist['spot_price'].shift(1))
    df_hist['rv_5d'] = df_hist['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
    df_hist['vrp'] = df_hist['m1_iv'] - df_hist['rv_5d']
    df_hist['vrp_col'] = np.where(df_hist['vrp'] > 0, '#00cc00', '#ff0000') 

    c1, c2 = st.columns(2)
    with c1:
        fig_slope = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['slope'], marker_color=df_hist['slope_col']))
        fig_slope.update_layout(title="<b>Near Term Structure (M2-M1)</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_slope, width="stretch")
    with c2:
        fig_std = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['std_pct'], marker_color=df_hist['std_col']))
        fig_std.update_layout(title="<b>Daily Straddle Change %</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_std, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig_vrp = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['vrp'], marker_color=df_hist['vrp_col']))
        fig_vrp.update_layout(title="<b>VRP Index (Edge)</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vrp, width="stretch")
    with c4:
        fig_skew = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['skew_index'], mode='lines', line=dict(color='#3498db', width=2), fill='tozeroy'))
        fig_skew.update_layout(title="<b>Skew Index</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_skew, width="stretch")

    # 5. Missing Charts Re-added
    c5, c6 = st.columns(2)
    with c5:
        fig_vvix = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['india_vix'], mode='lines', line=dict(color='#f1c40f', width=2)))
        fig_vvix.update_layout(title="<b>INDIA VIX</b>", template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vvix, width="stretch")
    with c6:
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

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: 
        # UPDATED: Only showing the date, time removed
        st.markdown(f"**{df_all.iloc[0]['timestamp'].strftime('%d %b %Y')}**")
        sel_date = st.date_input("Date", value=df_all['timestamp'].max().date())
    
    df_sel = df_all[df_all['timestamp'].dt.date <= sel_date]
    if df_sel.empty: st.stop()
    signals, ctx, curr = run_engine_live(df_sel)

    with c2: 
        st.markdown("""<h1 style='text-align: center; margin: 0; padding: 0; color: #ffc107; font-size: 42px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0px 0px 15px rgba(255, 193, 7, 0.4);'> VOLSTATE System </h1>""", unsafe_allow_html=True)
        st.markdown("""<h3 style='text-align: center; margin: 0; padding: 0; color: #aaa; font-size: 16px; font-weight: 400; letter-spacing: 1px;'>Volatility Regime & Carry Integrity System</h3>""", unsafe_allow_html=True)
        
    with c3: 
        st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
        if ctx['is_roll']: st.markdown('<span class="pill pill-yellow">ROLLOVER</span>', unsafe_allow_html=True)
        elif ctx['is_late']: st.markdown('<span class="pill pill-orange">LATE CYCLE</span>', unsafe_allow_html=True)
        else: st.markdown('<span class="pill pill-gray">MID CYCLE</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    tab_dash, tab_docs = st.tabs(["📊 DASHBOARD", "📘 DOCUMENTATION"])

    with tab_dash:
        render_dashboard(df_sel, signals, ctx, curr, df_all)
        st.markdown("<br><hr>", unsafe_allow_html=True)
        with st.expander("📂 Raw Database"):
            st.dataframe(df_all.style.format("{:.2f}", subset=['spot_price', 'm1_straddle', 'm1_iv', 'm2_iv', 'm3_iv', 'skew_index', 'india_vix']))

    with tab_docs:
        render_documentation_tab()

if __name__ == "__main__":
    main()

