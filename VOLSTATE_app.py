import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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

# --- CSS STYLING (DASHBOARD ONLY) ---
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
    .exec-box { background-color: #1e252e; border: 2px solid #444; border-radius: 10px; padding: 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
    .cis-score { font-size: 48px; font-weight: 900; color: #fff; display: flex; align-items: center; gap: 10px; }
    .status-badge { padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 20px; text-transform: uppercase; color: #000;}
    .delta-arrow { font-size: 24px; font-weight: bold; }

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
    
    .strat-box { background-color: #1e252e; border-left: 4px solid #555; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }
    .strat-header { font-size: 16px; font-weight: bold; color: #ddd; margin-bottom: 10px; text-transform: uppercase; }
    .strat-list { margin-bottom: 0; padding-left: 20px; color: #bbb; font-size: 14px; }
    .strat-list li { margin-bottom: 5px; }
    
    .cycle-highlight { color: #ddd; font-weight: bold; }
    .cycle-dim { color: #555; text-decoration: line-through; font-size: 0.9em;}
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
    disc = (abs(((curr['spot_price'] - prev['spot_price']) / prev['spot_price']) * 100) < 0.1 and iv_chg > 0.5)
    
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
    if len(rpv_df) < 4: return False, {}
    s, e = rpv_df["STRESS"].values, rpv_df["EXPANSION"].values
    slope = s[-1] - s[-3]
    accel = (s[-1] - s[-2]) > (s[-2] - s[-3])
    return (s[-1] > 0.20 and slope > 0.08 and accel and s[-1] > e[-1] * 0.6), {"stress_slope": slope, "stress_accel": accel}

def compute_cis_score(rpv, drift, stress_accel, std_pct, m1, m2):
    return np.clip(0.40*(rpv['COMPRESSION']+rpv['TRANSITION']) + 0.20*np.clip(-std_pct/0.25, -1, 1) + 0.15*np.clip((m2-m1)/1.0, -1, 1) - 0.50*rpv['STRESS'] - 0.30*np.clip(drift['STRESS']/0.10, 0, 1) - 0.20*(1.0 if stress_accel else 0.0), -1, 1)

def get_cis_status(score):
    if score > 0.35: return "SAFE", "#28a745"
    if score > 0.15: return "CAUTION", "#ffc107"
    if score > 0.0: return "DEFENSIVE", "#fd7e14"
    if score > -0.25: return "EXIT BIAS", "#dc3545"
    return "IMMEDIATE EXIT", "#dc3545"

def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 5: return None, None, df_c.iloc[-1]
    curr, prev, prev2 = df_c.iloc[-1], df_c.iloc[-2], df_c.iloc[-3]
    rpv, lhs = compute_rpv(curr, prev, prev2)
    dom = max(rpv, key=rpv.get)
    colors = {"COMPRESSION": "#28a745", "TRANSITION": "#ffc107", "EXPANSION": "#fd7e14", "STRESS": "#dc3545"}
    
    rpv_hist = compute_rpv_series(df_c.tail(15))
    drift = compute_rpv_drift(rpv_hist)
    pre_stress, ps_det = detect_pre_stress(rpv_hist)
    
    std_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    cis = compute_cis_score(rpv, drift, pre_stress, std_pct, curr['m1_iv'], curr.get('m2_iv', curr['m1_iv']))
    
    # Delta
    cis_delta = 0.0
    if len(rpv_hist) >= 4:
        prev_cis = compute_cis_score(rpv_hist.iloc[-4], compute_rpv_drift(rpv_hist.iloc[:-3]), detect_pre_stress(rpv_hist.iloc[:-3])[0], 
                                     ((df_c.iloc[-4]['m1_straddle'] - df_c.iloc[-5]['m1_straddle']) / df_c.iloc[-5]['m1_straddle']) * 100, 
                                     df_c.iloc[-4]['m1_iv'], df_c.iloc[-4].get('m2_iv', df_c.iloc[-4]['m1_iv']))
        cis_delta = cis - prev_cis

    dte = curr.get('m1_dte', 30)
    signals = {
        't1': (curr['m1_iv'] - prev['m1_iv'] > 0.2, "RISING" if curr['m1_iv'] - prev['m1_iv'] > 0.2 else "STABLE", f"{curr['m1_iv'] - prev['m1_iv']:+.2f}%"), 
        't2': (std_pct > -0.1, "STALLED" if std_pct > -0.1 else "DECAYING", f"{std_pct:+.2f}%"), 
        't4': (curr.get('m2_iv',0) - curr['m1_iv'] < 0, "INVERTED" if curr.get('m2_iv',0) - curr['m1_iv'] < 0 else "NORMAL", f"{curr.get('m2_iv',0) - curr['m1_iv']:.2f}"), 
        't5': (curr['skew_index'] - prev['skew_index'] > 0.3, "RISING" if curr['skew_index'] - prev['skew_index'] > 0.3 else "FLAT", f"{curr['skew_index'] - prev['skew_index']:+.2f}"), 
    }
    
    cycle = {
        "entry": ("✅ SAFE", "#28a745") if (cis >= 0.15) and (rpv['COMPRESSION'] + rpv['TRANSITION'] >= 0.55) else ("🛑 NO ENTRY", "#dc3545"),
        "harvest": ("⚠️ CAUTION", "#ffc107") if (drift['STRESS'] > 0.05) or (rpv['STRESS'] > 0.15) else ("✅ STABLE", "#28a745"),
        "exit": ("🚨 EXIT NOW", "#dc3545") if (cis < 0) or (rpv['STRESS'] >= 0.25) or pre_stress else ("HOLD", "#888")
    }
    
    ctx = {
        'regime': dom, 'color': colors.get(dom, "#888"), 'confidence': "HIGH" if rpv[dom] > 0.55 else "MEDIUM",
        'rpv': rpv, 'risk': derive_risk_posture(rpv), 'drivers': [k for k,v in lhs.items() if v[dom]>0.6], 'counterforces': [k for k,v in lhs.items() if v[dom]<0.3],
        'is_roll': dte >= 28, 'is_late': dte <= 7, 'dte': dte, 'cycle': cycle,
        'cis': {'score': cis, 'label': get_cis_status(cis)[0], 'color': get_cis_status(cis)[1], 'delta': cis_delta},
        'entry_bool': (cis >= 0.15) and (rpv['COMPRESSION'] + rpv['TRANSITION'] >= 0.55),
        'harvest_bool': (drift['STRESS'] > 0.05) or (rpv['STRESS'] > 0.15),
        'exit_bool': (cis < 0) or (rpv['STRESS'] >= 0.25) or pre_stress
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
    arrow = "<span class='delta-arrow' style='color: #28a745;'>↑</span>" if cis['delta'] > 0.001 else "<span class='delta-arrow' style='color: #dc3545;'>↓</span>" if cis['delta'] < -0.001 else ""
    
    st.markdown(f"""
    <div class="exec-box" style="border-color: {cis['color']};">
        <div><div style="font-size: 14px; color: #888;">EXECUTIVE COMMAND</div><div class="status-badge" style="background-color: {cis['color']};">{cis['label']}</div></div>
        <div style="text-align: right;"><div style="font-size: 14px; color: #888;">CARRY INTEGRITY SCORE</div><div class="cis-score">{cis['score']*100:.0f}% {arrow}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # RPV Banner
    st.markdown(f"""
    <div class="regime-box" style="background-color: {ctx['color']}15; border-color: {ctx['color']}80;">
        <div style="text-align: center; font-size: 11px; color: #888; letter-spacing: 1px;">MARKET STRUCTURE</div>
        <div class="regime-label" style="color: {ctx['color']};">{ctx['regime']} <span style='font-size: 12px; color: #aaa'>({ctx['confidence']})</span></div>
        <div class="rpv-bar">
            <div style="width: {ctx['rpv']['COMPRESSION']*100}%; background: #28a745;"></div>
            <div style="width: {ctx['rpv']['TRANSITION']*100}%; background: #ffc107;"></div>
            <div style="width: {ctx['rpv']['EXPANSION']*100}%; background: #fd7e14;"></div>
            <div style="width: {ctx['rpv']['STRESS']*100}%; background: #dc3545;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cycle Logic UI
    c = ctx['cycle']
    p1 = "<span class='cycle-highlight'>Edge > 55%</span>" if ctx['entry_bool'] else "<span class='cycle-dim'>Edge Low</span>"
    p2 = "<span class='cycle-highlight' style='color:#ffc107'>Drift > 0.05</span>" if ctx['harvest_bool'] else "<span class='cycle-dim'>Stable</span>"
    p3 = "<span class='cycle-highlight' style='color:#dc3545'>Stress > 0.25</span>" if ctx['exit_bool'] else "<span class='cycle-dim'>Safe</span>"

    st.markdown(f"""
    <div style="margin-bottom: 25px; padding: 15px; background: #161b22; border: 1px solid #444; border-radius: 8px;">
        <div style="display: flex; gap: 15px;">
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">ENTRY</div><div style="font-size: 16px; font-weight: bold; color: {c['entry'][1]};">{c['entry'][0]}</div><div style="font-size: 10px; color: #aaa;">{p1}</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">HARVEST</div><div style="font-size: 16px; font-weight: bold; color: {c['harvest'][1]};">{c['harvest'][0]}</div><div style="font-size: 10px; color: #aaa;">{p2}</div>
            </div>
            <div style="flex: 1; text-align: center; padding: 10px; background: #222; border-radius: 6px;">
                <div style="font-size: 11px; color: #888;">EXIT</div><div style="font-size: 16px; font-weight: bold; color: {c['exit'][1]};">{c['exit'][0]}</div><div style="font-size: 10px; color: #aaa;">{p3}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: s=signals['t1']; render_tile("FRONT STRESS (M1)", s[0], s[1], s[2])
    with c2: s=signals['t2']; render_tile("THETA EFFICIENCY", s[0], s[1], s[2])
    with c3: s=signals['t4']; render_tile("CARRY INSULATION", s[0], s[1], s[2], True)
    with c4: s=signals['t5']; render_tile("HEDGING PRESSURE", s[0], s[1], s[2])

    st.markdown(f"""<div class="mini-diag"><span>SPOT: {curr['spot_price']:.0f}</span><span>ATM IV: {curr['m1_iv']:.2f}%</span><span>STRADDLE: {curr['m1_straddle']:.0f}</span><span>DTE: {ctx['dte']}</span></div>""", unsafe_allow_html=True)

    # --- ANALYTICS SECTION ---
    st.markdown('<div class="section-header">📊 Analytics</div>', unsafe_allow_html=True)
    
    df_chart = df_selected.sort_values('timestamp').tail(60)
    
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

    # 2. Spot vs Straddle Chart
    fig_spot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['spot_price'], line=dict(color='#3498db', width=2), name="Spot"), secondary_y=False)
    fig_spot.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['m1_straddle'], line=dict(color='#e74c3c', width=2, dash='dot'), name="Straddle"), secondary_y=True)
    fig_spot.update_layout(title="<b>Nifty Spot vs ATM Straddle Price Trend</b>", template="plotly_dark", height=350, margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
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

    # 4. Metrics Charts (Slope, Std Pct, VRP, Skew)
    df_hist = df_chart.copy()
    df_hist['slope'] = df_hist['m3_iv'] - df_hist['m1_iv']
    df_hist['slope_col'] = np.where(df_hist['slope'] >= 0, '#00cc00', '#ff0000')
    df_hist['std_pct'] = df_hist['m1_straddle'].pct_change() * 100
    df_hist['std_col'] = np.where(df_hist['std_pct'] <= 0, '#00cc00', '#ff0000') 
    df_hist['log_ret'] = np.log(df_hist['spot_price'] / df_hist['spot_price'].shift(1))
    df_hist['rv_5d'] = df_hist['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
    df_hist['vrp'] = df_hist['m1_iv'] - df_hist['rv_5d']
    df_hist['vrp_col'] = np.where(df_hist['vrp'] > 0, '#00cc00', '#ff0000') 

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

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: 
        st.markdown(f"**{df_all.iloc[0]['timestamp'].strftime('%d %b %Y | %H:%M')} IST**")
        sel_date = st.date_input("Date", value=df_all['timestamp'].max().date())
    
    df_sel = df_all[df_all['timestamp'].dt.date <= sel_date]
    if df_sel.empty: st.stop()
    signals, ctx, curr = run_engine_live(df_sel)

    with c2: 
        st.markdown("""<h1 style='text-align: center; margin: 0; padding: 0; color: #ffc107; font-size: 42px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0px 0px 15px rgba(255, 193, 7, 0.4);'>⚡ VOLSTATE System </h1>""", unsafe_allow_html=True)
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
