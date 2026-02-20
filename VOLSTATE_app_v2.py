import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import textwrap
import importlib

# --- IMPORT DOCUMENTATION MODULE ---
try:
    import VOLSTATE_docs
    importlib.reload(VOLSTATE_docs)
    from VOLSTATE_docs import render_documentation_tab
except ImportError:
    def render_documentation_tab():
        st.error("VOLSTATE_docs.py not found. Please ensure the file exists in the same directory.")

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================
DB_NAME = "market_data_dummy.db"

# RPV & Expiry
RPV_EXPIRY_WINDOW_DAYS = 14

# Regime Stability Thresholds
REGIME_VELOCITY_STABLE = 0.02
REGIME_VELOCITY_DRIFT = 0.05

# Stress & Drift Settings
STRESS_DRIFT_LOOKBACK = 3
STRESS_DRIFT_THRESHOLD = 0.12
STRESS_DRIFT_NORM = 0.20
STRESS_DRIFT_PERSISTENCE = 2

# CIS Settings
CIS_STRESS_WEIGHT = -0.30
CIS_STRESS_DRIFT_WEIGHT = -0.20
CIS_STRESS_ACCEL_WEIGHT = -0.10
APPLY_STRESS_PENALTY_ONLY_IF_STRESS_GT = 0.18
CARRY_INSULATION_THRESHOLD = -0.25
CIS_FLOOR_COMPRESSION = 0.25
CIS_FLOOR_TRANSITION = 0.15

# RDE Settings
RDE_SLOPE_WINDOW = 3
MAX_DECAY_CAP = 2.5
RDE_WEAKENING_LOWER = 40
RDE_WEAKENING_UPPER = 60

# Page Config
st.set_page_config(
    page_title="VOLSTATE System",
    layout="wide",
    page_icon=" ⚡ ",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
#  CSS & UI HELPERS
# ==============================================================================
def inject_css():
    st.markdown("""
    <style>
        .main { font-family: 'Segoe UI', sans-serif; background-color: #0e1117; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e252e; border-radius: 5px; color: #fff; font-weight: 600; font-size: 16px; flex: 1; }
        .stTabs [aria-selected="true"] { background-color: #ffc107; color: #000; }
        
        /* Pills & Badges */
        .pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #fff; display: inline-block; }
        .pill-yellow { background-color: #ffc107; color: #000; }
        .pill-red { background-color: #dc3545; }
        .pill-orange { background-color: #fd7e14; }
        .pill-gray { background-color: #444; opacity: 0.5; }

        /* CIS Panel */
        .cis-box { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin-bottom: 25px; }
        .cis-strip { display: flex; justify-content: space-around; align-items: center; background-color: #161b22; border-radius: 4px; padding: 10px; border: 1px solid #21262d; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }
        .cis-metric { text-align: center; min-width: 110px; }
        .cis-val { font-size: 18px; font-weight: 700; color: #e6edf3; }
        .cis-lbl { font-size: 10px; color: #8b949e; text-transform: uppercase; margin-top: 4px; }
        .status-badge { padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 16px; text-transform: uppercase; color: #000; display: inline-block; }
        
        /* Tiles */
        .grid-tile { background-color: #161b22; border: 1px solid #333; border-radius: 6px; padding: 12px; height: 120px; display: flex; flex-direction: column; justify-content: space-between; position: relative; overflow: hidden; }
        .tile-header { font-size: 11px; color: #888; text-transform: uppercase; font-weight: 600; z-index: 2;}
        .tile-value { font-size: 22px; font-weight: 800; margin: 2px 0; z-index: 2;}
        .tile-sub { font-size: 11px; font-family: monospace; color: #aaa; z-index: 2;}
        .sparkline-container { position: absolute; bottom: 0; left: 0; right: 0; width: 100%; height: 40%; opacity: 0.35; z-index: 1; pointer-events: none; }
        
        /* Utility Colors */
        .text-green { color: #28a745; } .text-amber { color: #ffc107; } .text-red { color: #dc3545; } .text-gray { color: #888; }
        .border-green { border-left: 4px solid #28a745; } .border-amber { border-left: 4px solid #ffc107; } .border-red { border-left: 4px solid #dc3545; } .border-gray { border-left: 4px solid #555; }
        
        /* RDE & Dynamics */
        .rde-box { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin-top: 25px; margin-bottom: 25px; }
        .rde-strip { display: flex; justify-content: space-around; align-items: center; background-color: #161b22; border-radius: 4px; padding: 10px; border: 1px solid #21262d; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }
        .rde-metric { text-align: center; min-width: 100px; }
        .rde-val { font-size: 18px; font-weight: 700; color: #e6edf3; }
        .rde-lbl { font-size: 10px; color: #8b949e; text-transform: uppercase; margin-top: 4px; }
        .rde-badge { font-size: 14px; font-weight: bold; padding: 4px 8px; border-radius: 4px; text-transform: uppercase; }
        .rde-badge-green { background: #238636; color: white; }
        .rde-badge-amber { background: #bd5c00; color: white; }
        .rde-badge-red { background: #da3633; color: white; }
        .rde-badge-dis { background: #30363d; color: #8b949e; border: 1px solid #555; }
        .rde-badge-blue { background: #1f6feb; color: white; }
        .rde-badge-weak { background: #17a2b8; color: white; }

        .dynamics-console { background: linear-gradient(135deg, #161b22 0%, #1c2128 100%); border: 1px solid #555; border-radius: 8px; padding: 20px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 25px; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
        .ps-panel { flex: 1; min-width: 240px; border-right: 1px solid #444; padding-right: 20px; }
        .drift-grid { flex: 3; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 15px; }
        .drift-item { background: #0d1117; border: 1px solid #444; border-radius: 8px; padding: 12px; flex: 1; min-width: 100px; text-align: center; }
        .drift-val { font-size: 18px; font-weight: bold; margin-top: 4px; }
        
        .mini-diag { font-family: monospace; font-size: 12px; color: #666; border-top: 1px solid #333; margin-top: 20px; padding-top: 10px; display: flex; justify-content: space-around; flex-wrap: wrap; }
        .regime-box { text-align: center; padding: 15px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); background-color: rgba(255, 255, 255, 0.05); }
        .regime-label { font-size: 24px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
        .rpv-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 10px; width: 100%; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)

def render_metric_html(label, value, color=None):
    # Flattened single line to prevent markdown indentation issues
    style = f'style="color: {color}"' if color else ''
    return f'<div class="cis-metric"><div class="cis-val" {style}>{value}</div><div class="cis-lbl">{label}</div></div>'

def generate_sparkline_svg(data_series, color):
    if len(data_series) < 2: return ""
    min_val, max_val = min(data_series), max(data_series)
    rng = max_val - min_val if max_val != min_val else 1
    width, height = 200, 50
    step = width / (len(data_series) - 1)
    points = []
    coords = []
    for i, val in enumerate(data_series):
        x = i * step
        y = max(5, min(height - 5, height - ((val - min_val) / rng * height)))
        points.append(f"{x:.1f},{y:.1f}")
        coords.append((x, y))
    polyline = " ".join(points)
    dots_svg = "".join([f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}" />' for cx, cy in coords])
    return f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none"><polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>{dots_svg}</svg>'

def render_tile(label, state_bool, display_text, subtext, trend_data=None, is_stress=False):
    if state_bool is None: c, b, hex_c = "text-gray", "border-gray", "#888"
    elif state_bool: c, b, hex_c = ("text-red", "border-red", "#dc3545") if is_stress else ("text-amber", "border-amber", "#ffc107")
    else: c, b, hex_c = "text-green", "border-green", "#28a745"
    sparkline_html = f'<div class="sparkline-container">{generate_sparkline_svg(trend_data, hex_c)}</div>' if trend_data and len(trend_data) >= 2 else ""
    st.markdown(f'<div class="grid-tile {b}"><div class="tile-header">{label}</div><div class="tile-value {c}">{display_text}</div><div class="tile-sub">{subtext}</div>{sparkline_html}</div>', unsafe_allow_html=True)

# ==============================================================================
#  UTILITIES & DATA
# ==============================================================================
def load_data(limit=300):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql(f"SELECT * FROM market_logs ORDER BY timestamp DESC LIMIT {limit}", conn)
    except Exception as e:
        st.error(f"Database Error: {e}"); return pd.DataFrame()
    finally: conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure all required columns exist
        required_cols = ['m1_iv', 'm3_iv', 'm1_straddle', 'spot_price', 'm2_iv', 'skew_index', 
                         'india_vix', 'm2_straddle', 'cq_iv', 'nq_iv', 'cq_straddle', 'nq_straddle']
        for c in required_cols:
            if c not in df.columns: df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
        # Add Volatility Calculation
        df['log_ret'] = np.log(df['spot_price'] / df['spot_price'].shift(1))
        df['rv_5d'] = df['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
        df['rv_5d'] = df['rv_5d'].fillna(0.0)
            
    return df

def slope(series, lookback=3):
    """Unified slope calculation."""
    if len(series) <= lookback:
        return series.iloc[-1] - series.iloc[-2]
    return series.iloc[-1] - series.iloc[-(lookback+1)]

def pct_change(curr, prev):
    """Safe percentage change calculation."""
    if prev == 0: return 0
    return (curr - prev) / prev

def apply_expiry_gate(rpv, dte_m1, dSTR_M2, dIV_M2):
    """Unified expiry gating with normalization."""
    if dte_m1 <= RPV_EXPIRY_WINDOW_DAYS:
        m2_calm = (dSTR_M2 < 0) and (dIV_M2 <= 0)
        if m2_calm:
            suppressed = rpv['STRESS']
            rpv['STRESS'] = 0.0
            rpv['COMPRESSION'] += suppressed
            
            # Normalize
            total = sum(rpv.values())
            if total != 0:
                for k in rpv: rpv[k] /= total
    return rpv

def classify_regime_velocity(velocity):
    if abs(velocity) < REGIME_VELOCITY_STABLE: return "STABLE"
    elif abs(velocity) < REGIME_VELOCITY_DRIFT: return "DRIFTING"
    else: return "ACCELERATING"

def safe_neutral_state(curr_row):
    """Safety guard returning the actual row to prevent KeyErrors."""
    signals = {'t1': (False,"-","-",[]), 't2': (False,"-","-",[]), 't4': (False,"-","-",[]), 't5': (False,"-","-",[])}
    
    # Initialize defaults to prevent dashboard crashes
    default_drift = {r: 0.0 for r in REGIMES}
    default_ps_det = {"stress_slope": 0.0, "stress_accel": False}
    
    ctx = {
        'regime': "COMPRESSION", 'color': "#888", 'confidence': "LOW",
        'rpv': {"COMPRESSION":1.0, "TRANSITION":0, "EXPANSION":0, "STRESS":0},
        'is_roll': True, 'is_late': True, 'm1_dte': 0, 'm2_dte': 0,
        'cis': {'score': 0, 'label': "NEUTRAL", 'color': "#888", 'delta': 0},
        'cps': {'score': 0, 'label': "NEUTRAL", 'color': "#888"},
        'decay': {'val': 0, 'label': "SAFE", 'color': "#28a745"},
        'history': pd.DataFrame(), 
        'drift': default_drift, 
        'pre_stress': False, 
        'ps_det': default_ps_det,
        'expiry_badge': "EXPIRY HALT", 'regime_stability': "STABLE", 'divergence': {'msg': None, 'color': None}
    }
    return signals, ctx, curr_row

# ==============================================================================
#  CORE ENGINES: RPV, CIS, RDE
# ==============================================================================
REGIMES = ["COMPRESSION", "TRANSITION", "EXPANSION", "STRESS"]
WEIGHTS = {"iv": 1.2, "straddle": 1.2, "back_month": 1.0, "term": 1.0, "skew": 1.4, "disconnect": 0.8}

def iv_likelihood(iv_chg): return {"COMPRESSION": max(0, 1 - iv_chg/0.5), "TRANSITION": np.clip(iv_chg/0.6, 0, 1), "EXPANSION": np.clip(iv_chg/1.0, 0, 1), "STRESS": np.clip((iv_chg - 0.8)/1.2, 0, 1)}
def straddle_likelihood(std_pct): return {"COMPRESSION": 1 if std_pct < -0.2 else 0.2, "TRANSITION": np.clip((std_pct + 0.2)/0.4, 0, 1), "EXPANSION": np.clip((std_pct + 0.1)/0.6, 0, 1), "STRESS": np.clip((std_pct - 0.8)/1.0, 0, 1)}
def back_month_likelihood(bm_spread): return {"COMPRESSION": np.clip(-bm_spread / 0.3, 0, 1), "TRANSITION": np.clip(bm_spread/0.4, 0, 1), "EXPANSION": np.clip(bm_spread/0.6, 0, 1), "STRESS": np.clip(bm_spread/0.8, 0, 1)}
def term_likelihood(slope): return {"COMPRESSION": np.clip(slope/1.5, 0, 1), "TRANSITION": np.clip((1.2 - slope)/1.2, 0, 1), "EXPANSION": np.clip((0.8 - slope)/0.8, 0, 1), "STRESS": np.clip((-slope - 0.3)/0.7, 0, 1)}
def skew_likelihood(skew_chg): return {"COMPRESSION": 1 if skew_chg <= 0 else 0, "TRANSITION": np.clip((0.3 - skew_chg)/0.3, 0, 1), "EXPANSION": np.clip(skew_chg/0.6, 0, 1), "STRESS": np.clip((skew_chg - 0.6)/1.0, 0, 1)}
def disconnect_likelihood(disc): return {"COMPRESSION": 0, "TRANSITION": 0.4 if disc else 0.6, "EXPANSION": 0.7 if disc else 0.4, "STRESS": 0.9 if disc else 0.2}

def compute_rpv(curr, prev, prev2):
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    std_pct = pct_change(curr['m1_straddle'], prev['m1_straddle']) * 100
    bm_spread = (curr.get('m2_iv', 0) - prev.get('m2_iv', 0)) - iv_chg
    slope_val = curr['m3_iv'] - curr['m1_iv']
    skew_chg = curr['skew_index'] - prev['skew_index']
    disc = (abs(pct_change(curr['spot_price'], prev['spot_price']) * 100) < 0.15 and iv_chg > 0.5)

    lhs = {
        "iv": iv_likelihood(iv_chg), "straddle": straddle_likelihood(std_pct), "back_month": back_month_likelihood(bm_spread),
        "term": term_likelihood(slope_val), "skew": skew_likelihood(skew_chg), "disconnect": disconnect_likelihood(disc)
    }
    scores = {r: sum(WEIGHTS[k] * lhs[k][r] for k in lhs) for r in REGIMES}
    total = sum(scores.values())
    return {r: (scores[r]/total if total > 0 else 0) for r in REGIMES}, lhs

def compute_cis_score(rpv, drift, stress_accel, std_pct, m1, m2, regime):
    base = 0.40 * (rpv['COMPRESSION'] + rpv['TRANSITION'])
    theta_eff = 0.20 * np.clip(-std_pct/0.30, -0.8, 1)
    spread = m2 - m1
    insulation = -0.15 if spread < -0.25 else (+0.10 if spread > 0.25 else 0.0)
    
    if rpv['STRESS'] < APPLY_STRESS_PENALTY_ONLY_IF_STRESS_GT:
        p_stress = p_drift = p_accel = 0.0
    else:
        p_stress = CIS_STRESS_WEIGHT * rpv['STRESS']
        p_drift = CIS_STRESS_DRIFT_WEIGHT * np.clip(drift['STRESS']/STRESS_DRIFT_NORM, 0, 1)
        p_accel = CIS_STRESS_ACCEL_WEIGHT * (1.0 if stress_accel else 0.0)

    raw_cis = base + theta_eff + insulation + p_stress + p_drift + p_accel
    if regime == "COMPRESSION": raw_cis = max(raw_cis, CIS_FLOOR_COMPRESSION)
    elif regime == "TRANSITION": raw_cis = max(raw_cis, CIS_FLOOR_TRANSITION)
    return np.clip(raw_cis, -1, 1)

def compute_cps_score(rpv, std_pct, skew_accel_bool, m1, m2, m3):
    struct = 0.40 * (rpv['EXPANSION'] + rpv['STRESS'])
    gamma_hat = 0.25 * np.clip(std_pct / 0.30, -1, 1)
    term_hat = 0.15 * np.clip(-(m3 - m1)/1.0, 0, 1)
    lag_convexity = 0.10 * np.clip((m2 - m1) / 0.8, -1, 1)
    accel_hat = 0.10 * (1.0 if skew_accel_bool else 0.0)
    return np.clip(struct + gamma_hat + term_hat + lag_convexity + accel_hat, -1, 0.8)

def compute_rde_engine(df, cis_permission, regime):
    if len(df) < 5: return None
    curr = df.iloc[-1]
    
    # Inputs & Slopes using Unified Slope
    dSTR_M1 = slope(df['m1_straddle'], RDE_SLOPE_WINDOW)
    dSTR_M2 = slope(df['m2_straddle'], RDE_SLOPE_WINDOW)
    dIV_M1 = slope(df['m1_iv'], RDE_SLOPE_WINDOW)
    dIV_M2 = slope(df['m2_iv'], RDE_SLOPE_WINDOW)
    dIV_CQ = slope(df['cq_iv'], RDE_SLOPE_WINDOW)
    dIV_NQ = slope(df['nq_iv'], RDE_SLOPE_WINDOW)

    # DTE & Long Selection
    def get_dte(d_str):
        if pd.isna(d_str): return 0
        try: return (datetime.strptime(str(d_str), "%d-%b-%Y") - datetime.now()).days
        except: return 0
    
    dte_cq = get_dte(curr.get('cq_expiry'))
    dte_nq = get_dte(curr.get('nq_expiry'))
    dte_m2 = get_dte(curr.get('m2_month'))

    if dte_cq >= 80:
        active_long, long_expiry, dte_long = 'CQ', curr.get('cq_expiry'), dte_cq
        long_str_series = df['cq_straddle']
    else:
        active_long, long_expiry, dte_long = 'NQ', curr.get('nq_expiry'), dte_nq
        long_str_series = df['nq_straddle']

    # Decay Check
    raw_expected_decay = (curr.get(f'{active_long.lower()}_straddle', 0) / dte_long) if dte_long > 0 else 0
    expected_cq_decay = min(raw_expected_decay, MAX_DECAY_CAP)
    actual_cq_decay = abs(slope(long_str_series, RDE_SLOPE_WINDOW))

    # Sanity
    wrong_long = False
    sanity_msg = "OK"
    if dte_long < 75: wrong_long = True; sanity_msg = "CRITICAL: LOW DTE"
    elif (dte_long - dte_m2) < 30: wrong_long = True; sanity_msg = "SPREAD TOO NARROW"
    
    # Logic
    carry_thresh = 0.15
    regime_ok = regime in ["COMPRESSION", "TRANSITION"]
    rde_allowed = (regime_ok and cis_permission >= carry_thresh and dSTR_M1 < 0 and dSTR_M2 <= 0 and not wrong_long)
    rde_reduce = ((dIV_CQ - dIV_NQ < 0) or (actual_cq_decay > expected_cq_decay and dte_long <= 110))
    
    # Exit Hard
    m2_m1_slope = slope(df['m2_iv'] - df['m1_iv'], 3)
    cq_m2_slope = slope(df['cq_iv'] - df['m2_iv'], 3)
    flattening = (m2_m1_slope < 0) and (cq_m2_slope < 0)
    
    rde_exit_hard = (dIV_M1 < 0 and dIV_M2 < 0 and dIV_CQ < 0 and dIV_NQ < 0 and flattening)
    rde_exit_time = (dte_long <= 65)

    # Health
    carry_score = 15*(1 if dSTR_M1 < 0 else 0) + 15*(1 if dSTR_M2 <= 0 else 0)
    ins_score = 25 * max(0, min(1, (dIV_CQ - dIV_NQ) / 2.0))
    decay_score = 20 * (1 if actual_cq_decay <= expected_cq_decay else 0)
    dte_score = 15 * max(0, min(1, (dte_long - 60) / 60))
    penalty = -20 if rde_exit_hard else 0
    raw_health = 0 if wrong_long else (carry_score + ins_score + decay_score + dte_score + penalty)
    health_score = max(0, min(100, raw_health))

    RDE_WEAKENING = (health_score < RDE_WEAKENING_UPPER and health_score >= RDE_WEAKENING_LOWER and not rde_reduce and not (rde_exit_hard or rde_exit_time))

    # Status Helper
    def classify_rde_status():
        if wrong_long: return "DISABLED", "dis"
        if rde_exit_hard or rde_exit_time: return "EXIT", "red"
        if rde_reduce: return "REDUCE", "amber"
        if RDE_WEAKENING: return "WEAKENING", "weak"
        if rde_allowed: return "ALLOWED", "green"
        return "HOLD", "blue"
    
    status, s_color = classify_rde_status()

    return {
        'status': status, 's_color': s_color, 'health': int(health_score),
        'long_dte': dte_long, 'long_expiry': long_expiry, 'sanity': not wrong_long, 'sanity_msg': sanity_msg,
        'dSTR_M1': dSTR_M1, 'dSTR_M2': dSTR_M2, 'dIV_diff': dIV_CQ - dIV_NQ,
        'act_decay': actual_cq_decay, 'exp_decay': expected_cq_decay, 'flattening': flattening
    }

def generate_rpv_history(df):
    """Unified history builder."""
    rows = []
    if len(df) < 4: return pd.DataFrame()
    
    for i in range(3, len(df)):
        _c = df.iloc[i]
        _rpv, _ = compute_rpv(_c, df.iloc[i-1], df.iloc[i-2])
        
        # Unified Gating
        try:
            m1_date = pd.to_datetime(_c['m1_month'], format='%d-%b-%Y')
            dte = (m1_date - _c['timestamp']).days
        except: dte = 30
        
        # Patch 1B: Unified Slope Usage on Slices
        dSTR_M2 = slope(df['m2_straddle'].iloc[:i+1], 3)
        dIV_M2 = slope(df['m2_iv'].iloc[:i+1], 3)
        
        _rpv = apply_expiry_gate(_rpv, dte, dSTR_M2, dIV_M2)
        
        # Metrics
        _std_pct = pct_change(_c['m1_straddle'], df.iloc[i-1]['m1_straddle']) * 100
        _dom = max(_rpv, key=_rpv.get)
        _cis = compute_cis_score(_rpv, {'STRESS':0}, False, _std_pct, _c['m1_iv'], _c.get('m2_iv', _c['m1_iv']), _dom)
        _skew_accel = (_c['skew_index'] - df.iloc[i-1]['skew_index']) > (df.iloc[i-1]['skew_index'] - df.iloc[i-2]['skew_index'])
        _cps = compute_cps_score(_rpv, _std_pct, _skew_accel, _c['m1_iv'], _c.get('m2_iv', _c['m1_iv']), _c['m3_iv'])
        
        # Flatten RPV to fix KeyErrors
        row_data = {'timestamp': _c['timestamp'], 'cis': _cis, 'cps': _cps, 'rpv': _rpv}
        for k, v in _rpv.items(): row_data[k] = v
        rows.append(row_data)
        
    return pd.DataFrame(rows)

# ==============================================================================
#  MAIN APP LOGIC
# ==============================================================================
def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 5: return None, None, df_c.iloc[-1]
    curr, prev = df_c.iloc[-1], df_c.iloc[-2]

    # 1. Expiry Setup & Safety Guard
    try:
        # FIX: Ensure we use only DATE part to avoid negative DTE due to time differences
        current_date = pd.to_datetime(curr['timestamp']).normalize()
        m1_expiry_date = pd.to_datetime(curr['m1_month'], format='%d-%b-%Y').normalize()
        m2_expiry_date = pd.to_datetime(curr['m2_month'], format='%d-%b-%Y').normalize()
        
        m1_dte_raw = (m1_expiry_date - current_date).days
        m2_dte_raw = (m2_expiry_date - current_date).days
        
        # Floor at 0 for visual logic, but check raw for safety guard
        m1_dte = max(0, m1_dte_raw)
        m2_dte = max(0, m2_dte_raw)
        
        # FIX: Trigger safety guard ONLY if past expiry (buffer of -1)
        if m1_dte_raw < -1: return safe_neutral_state(curr)
    except: 
        m1_dte = 30; m2_dte = 60
    
    IS_M1_EXPIRY_WINDOW = (m1_dte <= RPV_EXPIRY_WINDOW_DAYS)

    # 2. Raw RPV
    rpv, lhs = compute_rpv(curr, prev, df_c.iloc[-3])

    # 3. Confirmation Gate (Unified Slope)
    dSTR_M2_check = slope(df_c['m2_straddle'], 3)
    dIV_M2_check = slope(df_c['m2_iv'], 3)
    rpv = apply_expiry_gate(rpv, m1_dte, dSTR_M2_check, dIV_M2_check)

    # 4. History (Unified Builder)
    df_hist = generate_rpv_history(df_c.tail(60))

    # 5. Downstream Metrics
    dom = max(rpv, key=rpv.get)
    colors = {"COMPRESSION": "#28a745", "TRANSITION": "#ffc107", "EXPANSION": "#fd7e14", "STRESS": "#dc3545"}
    
    # Optimized History Slicing
    rpv_hist_short = df_hist.tail(15) 
    
    def compute_drift(r_df, lookback=3):
        return {r: (r_df[r].iloc[-1] - r_df[r].iloc[-(lookback+1)]) if len(r_df) > lookback else 0.0 for r in REGIMES}
    
    drift = compute_drift(rpv_hist_short)
    
    # Pre-Stress Detector
    s, e = rpv_hist_short["STRESS"].values, rpv_hist_short["EXPANSION"].values
    if len(s) > 3:
        ps_slope = s[-1] - s[-3]
        ps_accel = (s[-1] - s[-2]) > (s[-2] - s[-3])
        pre_stress = (s[-1] > 0.20 and ps_slope > 0.08 and ps_accel and s[-1] > e[-1] * 0.6)
        ps_det = {"stress_slope": ps_slope, "stress_accel": ps_accel}
    else: pre_stress = False; ps_det = {"stress_slope": 0, "stress_accel": False}

    std_pct = pct_change(curr['m1_straddle'], prev['m1_straddle']) * 100
    skew_accel_bool = (curr['skew_index'] - prev['skew_index']) > (prev['skew_index'] - df_c.iloc[-3]['skew_index'])
    
    # Effective Drift
    drift_stress_prev = rpv_hist_short['STRESS'].iloc[-2] - rpv_hist_short['STRESS'].iloc[-5] if len(rpv_hist_short) >= 5 else 0
    eff_drift = drift.copy()
    eff_drift['STRESS'] = drift['STRESS'] if (drift['STRESS'] > STRESS_DRIFT_THRESHOLD and drift_stress_prev > STRESS_DRIFT_THRESHOLD) else 0.0

    cis = compute_cis_score(rpv, eff_drift, pre_stress, std_pct, curr['m1_iv'], curr.get('m2_iv', curr['m1_iv']), dom)
    cps = compute_cps_score(rpv, std_pct, skew_accel_bool, curr['m1_iv'], curr.get('m2_iv', curr['m1_iv']), curr['m3_iv'])

    # Append current if needed (using flattened structure)
    if not df_hist.empty and cis != df_hist.iloc[-1]['cis']:
        new_row_data = {'timestamp': curr['timestamp'], 'cis': cis, 'cps': cps, 'rpv': rpv}
        for k, v in rpv.items(): new_row_data[k] = v
        new_row = pd.DataFrame([new_row_data])
        df_hist = pd.concat([df_hist, new_row], ignore_index=True)

    # Context & Decay
    cis_vals = df_hist['cis'].values
    decay_val = np.clip(-(cis_vals[-1] - cis_vals[-4]) / 0.15, 0, 1) if len(cis_vals) >= 4 else 0.0
    
    def get_decay_status(d):
        if d < 0.25: return "#28a745", "STABLE"
        if d < 0.5: return "#ffc107", "DECAYING"
        if d < 0.75: return "#fd7e14", "DANGEROUS"
        return "#d9534f", "EXIT ZONE"
    decay_color, decay_label = get_decay_status(decay_val)

    # Regime Velocity (Corrected)
    if len(df_hist) > 1:
        prev_rpv_row = df_hist.iloc[-2]
        dom_prob = rpv[dom]
        prev_prob = prev_rpv_row[dom]
        regime_stability = classify_regime_velocity(dom_prob - prev_prob)
    else: regime_stability = "STABLE"

    # Context Labels
    def get_cis_band(c):
        if c > 0.35: return "FULL CARRY", "#28a745"
        if c > 0.15: return "CONTROLLED CARRY", "#ffc107"
        if c > -0.05: return "TOLERANCE ONLY", "#fd7e14"
        return "NO CARRY", "#d9534f"
    
    # Signals for tiles
    t1_hist = df_c['m1_iv'].tail(10).tolist()
    t2_hist = df_c['m1_straddle'].tail(10).tolist()
    t4_series = df_c['m2_iv'] - df_c['m1_iv']
    t4_hist = t4_series.tail(10).tolist()
    t5_hist = df_c['skew_index'].tail(10).tolist()
    
    m2_m1_spread = curr.get('m2_iv', 0) - curr['m1_iv']
    signals = {
        't1': (curr['m1_iv'] - prev['m1_iv'] > 0.2, "RISING" if curr['m1_iv'] - prev['m1_iv'] > 0.2 else "STABLE", f"{curr['m1_iv'] - prev['m1_iv']:+.2f}%", t1_hist),
        't2': (std_pct > -0.1, "STALLED" if std_pct > -0.1 else "DECAYING", f"{std_pct:+.2f}%", t2_hist),
        't4': (m2_m1_spread < CARRY_INSULATION_THRESHOLD, "INVERTED" if m2_m1_spread < CARRY_INSULATION_THRESHOLD else "NORMAL", f"{m2_m1_spread:.2f}", t4_hist),
        't5': (curr['skew_index'] - prev['skew_index'] > 0.3, "RISING" if curr['skew_index'] - prev['skew_index'] > 0.3 else "FLAT", f"{curr['skew_index'] - prev['skew_index']:+.2f}", t5_hist),
    }
    
    band_label, band_color = get_cis_band(cis)

    # Build Context
    ctx = {
        'regime': dom, 'color': colors.get(dom, "#888"), 'confidence': "HIGH" if rpv[dom] > 0.55 else "MEDIUM",
        'rpv': rpv, 'is_roll': m1_dte >= 28, 'is_late': m1_dte <= 7, 'm1_dte': m1_dte, 'm2_dte': m2_dte,
        'cis': {'score': cis, 'label': band_label, 'color': band_color, 'delta': cis - (cis_vals[-2] if len(cis_vals)>1 else cis)},
        'cps': {'score': cps, 'label': "NEUTRAL", 'color': "#888"},
        'decay': {'val': decay_val, 'label': decay_label, 'color': decay_color},
        'history': df_hist, 'drift': eff_drift, 'pre_stress': pre_stress, 'ps_det': ps_det,
        'expiry_badge': "EXPIRY WINDOW" if IS_M1_EXPIRY_WINDOW else None,
        'regime_stability': regime_stability,
        'divergence': {'msg': None, 'color': None}
    }
    return signals, ctx, curr

def render_cis_ui(ctx, curr, prev):
    cis = ctx['cis']
    rpv = ctx['rpv']
    
    if cis['score'] >= 0.35: c_stat, c_col = "ALLOWED", "#28a745"
    elif cis['score'] >= 0.15: c_stat, c_col = "CAUTION", "#ffc107"
    else: c_stat, c_col = "BLOCKED", "#d9534f"
    
    c_arrow = "↑" if cis['delta'] > 0.001 else "↓" if cis['delta'] < -0.001 else "→"
    
    curr_vrp = curr['m1_iv'] - curr.get('rv_5d', 0)
    prev_vrp = prev['m1_iv'] - prev.get('rv_5d', 0)
    vrp_arrow = "↑" if curr_vrp > prev_vrp else "↓" if curr_vrp < prev_vrp else ""
    vrp_col = "#28a745" if curr_vrp > 1.0 else ("#ffc107" if curr_vrp > 0.2 else "#d9534f")

    std_pct = pct_change(curr['m1_straddle'], prev['m1_straddle']) * 100
    theta_arrow = "↓" if std_pct < -0.05 else "↑" if std_pct > 0.05 else ""
    theta_col = "#28a745" if std_pct < -0.25 else ("#ffc107" if std_pct < -0.10 else "#d9534f")

    s_val = rpv['STRESS']
    s_col = "#28a745" if s_val < 0.15 else ("#ffc107" if s_val < 0.25 else "#d9534f")

    p_val = rpv[ctx['regime']] * 100
    p_col = "#28a745" if p_val > 55 else ("#ffc107" if p_val > 45 else "#d9534f")
    
    iv_chg = curr['m1_iv'] - prev['m1_iv']
    iv_trend = "BLEEDING" if iv_chg < -0.2 else ("BIDDING" if iv_chg > 0.2 else "STABLE")

    # FIXED: Direct list join to construct HTML without indentation issues
    html_parts = [
        '<div class="cis-box">',
        '<div style="font-size: 11px; color: #58a6ff; font-weight:bold; letter-spacing:1px; margin-bottom:10px;">CARRY INTEGRITY PANEL</div>',
        '<div class="cis-strip">',
        f'<div class="cis-metric"><div class="status-badge" style="background:{c_col};">{c_stat}</div><div class="cis-lbl">CARRY STATUS</div></div>',
        render_metric_html("CARRY SCORE", f"{int(cis['score']*100)}% {c_arrow}"),
        render_metric_html("REGIME", ctx['regime']),
        render_metric_html("IV TREND", iv_trend),
        '</div>',
        '<div class="cis-strip">',
        render_metric_html("Vol Premium", f"{curr_vrp:+.1f}% {vrp_arrow}", vrp_col),
        render_metric_html("Time Decay", f"{std_pct:+.2f}% {theta_arrow}", theta_col),
        render_metric_html("Market Stress", f"{s_val:.2f}", s_col),
        render_metric_html("Regime Stickiness", f"{int(p_val)}%", p_col),
        '</div>',
        f'<div style="text-align: center; font-size: 12px; color: #aaa; margin-top: 5px;">Regime Stability: <span style="font-weight: bold; color: #e6edf3;">{ctx["regime_stability"]}</span></div>',
        '</div>'
    ]
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def render_rde_panel(rde, history):
    if not rde: return
    c_col = f"rde-badge-{rde['s_color']}"
    
    # FIXED: Direct list join
    html_parts = [
        '<div class="rde-box">',
        '<div style="font-size: 11px; color: #58a6ff; font-weight:bold; letter-spacing:1px; margin-bottom:10px;">RDE CONTROL PANEL (REGIME DISCIPLINE)</div>',
        '<div class="rde-strip">',
        f'<div class="rde-metric"><div class="rde-badge {c_col}">{rde["status"]}</div><div class="rde-lbl">RDE STATUS</div></div>',
        render_metric_html("HEALTH SCORE", rde['health'], '#238636' if rde['health']>70 else '#e67e22'),
        render_metric_html("LONG DTE", f"{rde['long_dte']} d", '#e6edf3' if rde['long_dte']>90 else '#da3633'),
        render_metric_html("CONTRACT", rde['long_expiry']),
        render_metric_html(rde['sanity_msg'], '✅' if rde['sanity'] else '❌'),
        '</div>'
    ]
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    carry_ok = not (rde['dSTR_M1'] >= 0 or rde['dSTR_M2'] > 0)
    is_stress = rde['dSTR_M2'] > 0
    with c1: render_tile("CARRY INTEGRITY", not carry_ok, "STABLE" if carry_ok else "DEGRADING", f"M1: {rde['dSTR_M1']:.1f}", history['dSTR_M1'], is_stress)
    
    lead_warn = rde['dIV_diff'] < 0
    with c2: render_tile("INS. LEADERSHIP", lead_warn, f"{rde['dIV_diff']:+.2f}", "CQ - NQ Slope", history['dIV_diff'], True)
    
    dec_warn = rde['act_decay'] > rde['exp_decay']
    with c3: render_tile("QUARTERLY DECAY", dec_warn, f"{rde['act_decay']:.1f}", f"Exp: {rde['exp_decay']:.1f}", history['act_decay'], True)
    
    risk_warn = rde['flattening']
    with c4: render_tile("STRUCTURAL RISK", risk_warn, "FLATTENING" if risk_warn else "NORMAL", "Curve Shape", history['struct_risk'], True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_dashboard(df_selected, signals, ctx, curr, df_all):
    # Regime Box
    p_comp, p_tran = int(ctx['rpv']['COMPRESSION'] * 100), int(ctx['rpv']['TRANSITION'] * 100)
    p_expa, p_strs = int(ctx['rpv']['EXPANSION'] * 100), int(ctx['rpv']['STRESS'] * 100)
    badge_html = f"<div style='margin-bottom: 5px;'><span class='pill pill-red'>{ctx['expiry_badge']}</span></div>" if ctx['expiry_badge'] else ""
    
    st.markdown(f"""
    <div class="regime-box" style="background-color: {ctx['color']}15; border-color: {ctx['color']}80;">
        <div style="text-align: center; font-size: 11px; color: #888; letter-spacing: 1px;">MARKET STRUCTURE</div>
        {badge_html}
        <div class="regime-label" style="color: {ctx['color']};">{ctx['regime']} <span style='font-size: 12px; color: #aaa'>({ctx['confidence']})</span></div>
        <div class="rpv-bar">
            <div style="width: {ctx['rpv']['COMPRESSION']*100}%; background: #28a745;"></div>
            <div style="width: {ctx['rpv']['TRANSITION']*100}%; background: #ffc107;"></div>
            <div style="width: {ctx['rpv']['EXPANSION']*100}%; background: #fd7e14;"></div>
            <div style="width: {ctx['rpv']['STRESS']*100}%; background: #dc3545;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; font-family: monospace; font-weight: bold;">
            <div style="color: #28a745;">COMP: {p_comp}%</div><div style="color: #ffc107;">TRAN: {p_tran}%</div>
            <div style="color: #fd7e14;">EXPA: {p_expa}%</div><div style="color: #dc3545;">STRS: {p_strs}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CIS Panel
    render_cis_ui(ctx, curr, df_selected.iloc[-2])

    # Tiles
    c1, c2, c3, c4 = st.columns(4)
    with c1: s=signals['t1']; render_tile("FRONT STRESS (M1)", s[0], s[1], s[2], s[3], True)
    with c2: s=signals['t2']; render_tile("THETA EFFICIENCY", s[0], s[1], s[2], s[3])
    with c3: s=signals['t4']; render_tile("CARRY INSULATION", s[0], s[1], s[2], s[3], True)
    with c4: s=signals['t5']; render_tile("HEDGING PRESSURE", s[0], s[1], s[2], s[3], True)
    
    st.markdown(f"""<div class="mini-diag">
    <span>SPOT: {curr['spot_price']:.0f}</span><span>ATM IV: {curr['m1_iv']:.2f}%</span>
    <span>STRADDLE: {curr['m1_straddle']:.0f}</span><span>M1 DTE: {ctx['m1_dte']}</span><span>M2 DTE: {ctx['m2_dte']}</span>
    </div>""", unsafe_allow_html=True)

    # RDE Panel
    df_chart = df_selected.sort_values('timestamp').tail(60)
    df_chart['dSTR_M1'] = df_chart['m1_straddle'].diff(3)
    df_chart['dIV_diff'] = df_chart['cq_iv'].diff(3) - df_chart['nq_iv'].diff(3)
    df_chart['act_decay'] = df_chart['cq_straddle'].diff(3).abs()
    df_chart['struct_risk'] = df_chart['m2_iv'] - df_chart['m1_iv']
    
    rde_hist = {k: df_chart[k].tail(10).tolist() for k in ['dSTR_M1', 'dIV_diff', 'act_decay', 'struct_risk']}
    rde_data = compute_rde_engine(df_selected, ctx['cis']['score'], ctx['regime'])
    render_rde_panel(rde_data, rde_hist)

    # Dynamics
    ps_status = "<span class='ps-status-danger'> ⚠️ TRIGGERED</span>" if ctx['pre_stress'] else "<span class='ps-status-safe'> ✅ SAFE</span>"
    def fmt_drift(val):
        c = "#00e676" if val > 0.1 else ("#28a745" if val > 0.05 else ("#dc3545" if val < -0.05 else "#888"))
        return f"<span style='color:{c}; font-weight:bold;'>{'↑' if val>0 else '↓'} {abs(val):.2f}</span>"
    
    st.markdown(f"""
    <div class="dynamics-console">
        <div class="ps-panel">
            <div class="dynamics-title">PRE-STRESS DETECTOR</div>
            <div class="ps-status">{ps_status}</div>
            <div class="ps-metrics">
                <div>Slope: {ctx['ps_det']['stress_slope']:.2f}</div>
                <div>Accel: {ctx['ps_det']['stress_accel']}</div>
            </div>
        </div>
        <div class="drift-grid">
            <div class="drift-item"><div class="drift-label">COMP DRIFT</div><div class="drift-val">{fmt_drift(ctx['drift']['COMPRESSION'])}</div></div>
            <div class="drift-item"><div class="drift-label">TRAN DRIFT</div><div class="drift-val">{fmt_drift(ctx['drift']['TRANSITION'])}</div></div>
            <div class="drift-item"><div class="drift-label">EXPA DRIFT</div><div class="drift-val">{fmt_drift(ctx['drift']['EXPANSION'])}</div></div>
            <div class="drift-item"><div class="drift-label">STRS DRIFT</div><div class="drift-val">{fmt_drift(ctx['drift']['STRESS'])}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-header"> 📊 Analytics</div>', unsafe_allow_html=True)
    fig_spot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['spot_price'], line=dict(color='#3498db', width=2), name="Spot"), secondary_y=False)
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['m1_straddle'], line=dict(color='#e74c3c', width=2, dash='dot'), name="Straddle"), secondary_y=True)
    fig_spot.update_layout(title="<b>Nifty Spot vs ATM Straddle Price Trend</b>", template="plotly_dark", height=350, margin=dict(t=20,b=20,l=20,r=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_spot, width="stretch")

    rpv_hist_full = generate_rpv_history(df_selected.tail(60))
    if not rpv_hist_full.empty:
        fig_rpv = go.Figure()
        for r, c in [('STRESS','#dc3545'), ('EXPANSION','#fd7e14'), ('TRANSITION','#ffc107'), ('COMPRESSION','#28a745')]:
            fig_rpv.add_trace(go.Scatter(x=rpv_hist_full['timestamp'], y=rpv_hist_full[r], mode='lines', stackgroup='one', name=r, line=dict(color=c, width=0)))
        fig_rpv.update_layout(title="<b>Regime Probabilities (2 Months)</b>", template="plotly_dark", height=300, margin=dict(t=40,b=10,l=10,r=10), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_rpv, width="stretch")

def main():
    inject_css()
    df_all = load_data(300)
    if len(df_all) < 5: st.error(" ⚠️ Not enough data found."); st.stop()
    
    c1, c2, c3 = st.columns([1, 6, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"**{df_all.iloc[0]['timestamp'].strftime('%d %b %Y')}**")
        sel_date = st.date_input("Date", value=df_all['timestamp'].max().date())
    
    df_sel = df_all[df_all['timestamp'].dt.date <= sel_date]
    if df_sel.empty: st.stop()
    
    signals, ctx, curr = run_engine_live(df_sel)
    
    with c2:
        st.markdown("<div style='text-align: center;'><h1 style='margin:0; padding:0; color:#ffc107; font-size:42px; font-weight:900; letter-spacing:2px; text-shadow:0px 0px 15px rgba(255,193,7,0.4);'>VOL ⚡ TATE System</h1><h3 style='margin:0; color:#aaa; font-size:16px;'>Volatility Regime & Carry Integrity System</h3></div>", unsafe_allow_html=True)
    with c3:
        tag = "ROLLOVER" if ctx['is_roll'] else ("LATE CYCLE" if ctx['is_late'] else "MID CYCLE")
        cls = "pill-yellow" if ctx['is_roll'] else ("pill-orange" if ctx['is_late'] else "pill-gray")
        st.markdown(f'<div style="text-align: right; margin-top: 10px;"><span class="pill {cls}">{tag}</span></div>', unsafe_allow_html=True)

    tab_dash, tab_docs = st.tabs([" 📊 DASHBOARD", " 📘 DOCUMENTATION"])
    with tab_dash:
        render_dashboard(df_sel, signals, ctx, curr, df_all)
        st.markdown("<br><hr>", unsafe_allow_html=True)
        with st.expander(" 📂 Raw Database"):
            st.dataframe(df_all.style.format("{:.2f}", subset=['spot_price', 'm1_straddle', 'm1_iv', 'm2_iv', 'm3_iv', 'skew_index', 'india_vix']))
    with tab_docs:
        render_documentation_tab()

if __name__ == "__main__":
    main()