import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- CONFIG ---
DB_NAME = "market_data.db"

# --- STRATEGY MATRIX DATA ---
STRATEGY_MATRIX = {
    "COMPRESSION": {
        "works": ["Time decay (theta capture)", "Volatility mean reversion", "Range-bound price action", "Stable contango"],
        "strategies": ["Short ATM / near-ATM straddles or strangles", "Iron fly / iron condor", "Credit spreads (call or put)", "Ratio spreads with decay bias"]
    },
    "TRANSITION": {
        "works": ["Reduced exposure", "Flexibility over optimisation", "Structures that survive chop", "Limited downside"],
        "strategies": ["Wide iron condor (small size)", "Defined-risk credit spreads", "Calendar / diagonal spreads", "Partial hedged carry"]
    },
    "EXPANSION": {
        "works": ["Volatility expansion", "Convexity (gamma + vega)", "Directional follow-through", "Breakout / trend behaviour"],
        "strategies": ["Long straddles / strangles", "Debit spreads", "Back-spread structures", "Calendars tilted long vol"]
    },
    "STRESS": {
        "works": ["Capital preservation", "Asymmetry", "Tail protection", "Optionality over income"],
        "strategies": ["Long puts / call spreads (hedges)", "Very small long-vol positions", "Synthetic protection", "No trade / flat book"]
    }
}

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
    .regime-box { text-align: center; padding: 15px; border-radius: 8px; margin-bottom: 25px; border: 1px solid #333; }
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
    .border-green { border-left: 4px solid #28a745; }
    .border-amber { border-left: 4px solid #ffc107; }
    .border-red { border-left: 4px solid #dc3545; }
    .mini-diag { font-family: monospace; font-size: 12px; color: #666; border-top: 1px solid #333; margin-top: 20px; padding-top: 10px; display: flex; justify-content: space-around; }
    .section-header { margin-top: 40px; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #333; font-size: 20px; font-weight: bold; color: #ddd;}
    /* Hide Streamlit Date Input Label if desired, or style it */
    .stDateInput label { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADERS ---
def load_data(limit=300): # Increased limit to allow scrolling back in history
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
        
        cols_to_numeric = ['m1_iv', 'm3_iv', 'm1_straddle', 'spot_price', 'm2_iv', 'skew_index', 'vvix']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
    return df

# --- UI HELPER: RENDER TILE ---
def render_tile(label, state_bool, display_text, subtext, is_stress=False):
    if state_bool:
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

# --- LOGIC ENGINE ---
def run_engine_single_row(curr, prev, prev2):
    if prev is None or prev2 is None: return None, None, 0

    iv_chg_1 = curr['m1_iv'] - prev['m1_iv']
    iv_chg_2 = prev['m1_iv'] - prev2['m1_iv']
    iv_rising_count = sum([1 for x in [iv_chg_1, iv_chg_2] if x > 0])
    
    straddle_chg_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    spot_chg_pct = ((curr['spot_price'] - prev['spot_price']) / prev['spot_price']) * 100
    
    m2_curr = curr.get('m2_iv', 0)
    m2_prev = prev.get('m2_iv', 0)
    back_month_spread_chg = (m2_curr - m2_prev) - iv_chg_1
    
    slope = curr['m3_iv'] - curr['m1_iv']
    m2_iv = curr.get('m2_iv', (curr['m1_iv'] + curr['m3_iv'])/2)
    
    skew_chg = curr['skew_index'] - prev['skew_index']
    is_disconnect = (spot_chg_pct > 0.2 and iv_chg_1 > 0.2) or (abs(spot_chg_pct) < 0.1 and iv_chg_1 > 0.5)

    dte = curr.get('m1_dte', 30)
    is_rollover_window = dte >= 28 
    is_late_cycle = dte <= 7

    t1_state = (iv_rising_count >= 2 and iv_chg_1 > 0.4) or (iv_chg_1 > 1.0)
    
    if is_late_cycle:
        t2_state = straddle_chg_pct > -0.05
    else:
        t2_state = straddle_chg_pct > -0.2
    
    t3_state = False if is_rollover_window else (back_month_spread_chg > 0.4)
    
    if is_rollover_window: 
        t4_state = False
    elif slope < 0: 
        t4_state = True 
    elif (m2_iv - curr['m1_iv']) >= 0.6 and (curr['m3_iv'] - m2_iv) <= 0.2:
        t4_state = True
    elif slope < 1.0: 
        t4_state = True
    else: 
        t4_state = False
        
    t5_state = skew_chg > 0.5
    t6_state = is_disconnect
    
    yes_count = sum([t1_state, t2_state, t3_state, t4_state, t5_state, t6_state])
    
    is_stress = (
        slope < -0.5 
        or (straddle_chg_pct > 1.0 and abs(spot_chg_pct) < 0.4) 
        or (t5_state and t1_state)
    )

    if is_stress: 
        regime = "STRESS"
        color = "#dc3545"
    elif yes_count >= 3: 
        regime = "EXPANSION"
        color = "#fd7e14"
    elif yes_count >= 2: 
        regime = "TRANSITION"
        color = "#ffc107"
    else: 
        regime = "COMPRESSION"
        color = "#28a745"
    
    return regime, color, yes_count

def run_engine_live(df):
    df_c = df.sort_values('timestamp', ascending=True).copy()
    if len(df_c) < 3: return None, None, df_c.iloc[-1]

    curr = df_c.iloc[-1]
    prev = df_c.iloc[-2]
    prev2 = df_c.iloc[-3]
    
    regime, color, count = run_engine_single_row(curr, prev, prev2)
    
    dte = curr.get('m1_dte', 30)
    is_rollover = dte >= 28
    is_late_cycle = dte <= 7
    is_event = False 
    
    iv_chg_1 = curr['m1_iv'] - prev['m1_iv']
    iv_chg_2 = prev['m1_iv'] - prev2['m1_iv']
    iv_rising_count = sum([1 for x in [iv_chg_1, iv_chg_2] if x > 0])
    
    straddle_chg_pct = ((curr['m1_straddle'] - prev['m1_straddle']) / prev['m1_straddle']) * 100
    spot_chg_pct = ((curr['spot_price'] - prev['spot_price']) / prev['spot_price']) * 100
    
    m2_iv_curr = curr.get('m2_iv', 0)
    m2_iv_prev = prev.get('m2_iv', 0)
    m2_chg = m2_iv_curr - m2_iv_prev
    back_month_spread_chg = m2_chg - iv_chg_1
    
    slope = curr['m3_iv'] - curr['m1_iv']
    m2_iv = curr.get('m2_iv', (curr['m1_iv'] + curr['m3_iv'])/2)
    
    skew_chg = curr['skew_index'] - prev['skew_index']
    is_disconnect = (spot_chg_pct > 0.2 and iv_chg_1 > 0.2) or (abs(spot_chg_pct) < 0.1 and iv_chg_1 > 0.5)

    t1_state = (iv_rising_count >= 2 and iv_chg_1 > 0.4) or (iv_chg_1 > 1.0)
    t1_text = "YES" if t1_state else "NO"
    t1_sub = f"{iv_chg_1:+.2f} vol | {iv_rising_count}/2 days"
    
    if is_late_cycle:
        t2_state = straddle_chg_pct > -0.05
    else:
        t2_state = straddle_chg_pct > -0.2
    t2_text = "STALLED" if t2_state else "DECAYING"
    t2_sub = f"Std {straddle_chg_pct:+.2f}% | Spot {spot_chg_pct:+.2f}%"
    
    if is_rollover:
        t3_state = False; t3_text = "GATED"; t3_sub = "Rollover Window"
    else:
        t3_state = back_month_spread_chg > 0.4
        t3_text = "YES" if t3_state else "NO"
        t3_sub = f"Next {m2_chg:+.2f} | Near {iv_chg_1:+.2f}"
        
    if is_rollover:
        t4_state = False; t4_text = "GATED"
    elif slope < 0:
        t4_state = True; t4_text = "INVERTED"
    elif (m2_iv - curr['m1_iv']) >= 0.6 and (curr['m3_iv'] - m2_iv) <= 0.2:
        t4_state = True; t4_text = "KINKED"
    elif slope < 1.0:
        t4_state = True; t4_text = "FLAT"
    else:
        t4_state = False; t4_text = "STABLE"
    t4_sub = f"Slope: {slope:.2f} pts"
    
    t5_state = skew_chg > 0.5
    t5_text = "YES" if t5_state else "NO"
    t5_sub = f"Skew Idx {skew_chg:+.2f}"
    
    t6_state = is_disconnect
    t6_text = "YES" if t6_state else "NO"
    t6_sub = f"Spot {spot_chg_pct:+.1f}% | IV {iv_chg_1:+.1f}"

    if is_late_cycle and regime == "COMPRESSION":
        regime = "COMPRESSION (LATE)"

    if count <= 1: confidence = "HIGH"
    elif count == 2: confidence = "MEDIUM"
    else: confidence = "LOW (UNSTABLE)"

    signals = {
        't1': (t1_state, t1_text, t1_sub), 't2': (t2_state, t2_text, t2_sub),
        't3': (t3_state, t3_text, t3_sub), 't4': (t4_state, t4_text, t4_sub),
        't5': (t5_state, t5_text, t5_sub), 't6': (t6_state, t6_text, t6_sub),
    }
    
    context = {
        'regime': regime, 'color': color, 'count': count, 'confidence': confidence,
        'is_roll': is_rollover, 'is_late': is_late_cycle, 'is_event': is_event, 'dte': dte
    }
    return signals, context, curr

def calculate_historical_regime(df):
    df_sorted = df.sort_values('timestamp', ascending=True).copy()
    history = []
    
    for i in range(2, len(df_sorted)):
        curr = df_sorted.iloc[i]
        prev = df_sorted.iloc[i-1]
        prev2 = df_sorted.iloc[i-2]
        
        regime, color, _ = run_engine_single_row(curr, prev, prev2)
        history.append({'timestamp': curr['timestamp'], 'regime': regime, 'color': color, 'val': 1})
        
    return pd.DataFrame(history)

# --- MAIN APP ---
def main():
    # Load more data to ensure we have history for scrolling
    df_all = load_data(300) 
    
    if len(df_all) < 5:
        st.error("⚠️ Not enough data found. Need at least 5 days of history.")
        st.stop()

    # --- TOP HEADER ---
    c_head1, c_head2, c_head3 = st.columns([1, 2, 1])
    
    # 1. Date Picker Logic
    max_date = df_all['timestamp'].max().date()
    min_date = df_all['timestamp'].min().date()

    with c_head1:
        st.markdown(f"**{df_all.iloc[0]['timestamp'].strftime('%d %b %Y | %H:%M')} IST**")
        # Go to particular day option
        selected_date = st.date_input("Go to Date:", value=max_date, min_value=min_date, max_value=max_date)

    # Filter Data for Engine based on Selected Date
    # We slice the DF to pretend 'selected_date' is the last available day
    df_selected = df_all[df_all['timestamp'].dt.date <= selected_date]
    
    if df_selected.empty:
        st.error("No data available for selected date.")
        st.stop()

    # Run engine on the filtered slice
    signals, ctx, curr = run_engine_live(df_selected)

    with c_head2:
        # UPDATED: Added Emoji ⚡
        st.markdown("""
            <h1 style='text-align: center; margin: 0; padding: 0; color: #ffc107; 
            font-size: 42px; font-weight: 900; text-transform: uppercase; 
            letter-spacing: 2px; text-shadow: 0px 0px 15px rgba(255, 193, 7, 0.4);'>
            ⚡ VOLSTATE DASHBOARD</h1>
            """, unsafe_allow_html=True)
        
    with c_head3:
        st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
        if ctx['is_roll']: 
            st.markdown('<span class="pill pill-yellow">🟡 ROLLOVER WINDOW</span>', unsafe_allow_html=True)
        elif ctx['is_late']: 
            st.markdown('<span class="pill pill-orange">🟠 LATE CYCLE</span>', unsafe_allow_html=True)
        else:
             st.markdown('<span class="pill pill-gray">MID CYCLE</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="regime-box" style="background-color: {ctx['color']}20; border-color: {ctx['color']};">
            <div class="regime-sub" style="color: {ctx['color']}">DETECTED REGIME</div>
            <div class="regime-label" style="color: {ctx['color']}">{ctx['regime']}</div>
            <div class="regime-sub">CONFIRMATION: {ctx['count']}/6 SIGNALS</div>
            <div class="regime-sub" style="font-weight: bold; margin-top: 5px;">CONFIDENCE: {ctx['confidence']}</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        s = signals['t1']
        render_tile("ATM IV RISING", s[0], s[1], s[2])
    with c2:
        s = signals['t2']
        render_tile("STRADDLE STALLED?", s[0], s[1], s[2])
    with c3:
        s = signals['t3']
        render_tile("BACK MONTHS BID", s[0], s[1], s[2])

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        s = signals['t4']
        render_tile("TERM STRUCTURE", s[0], s[1], s[2], is_stress=True)
    with c5:
        s = signals['t5']
        render_tile("TAIL RISK (SKEW)", s[0], s[1], s[2])
    with c6:
        s = signals['t6']
        render_tile("VOL vs SPOT DISCONNECT", s[0], s[1], s[2])

    st.markdown(f"""
        <div class="mini-diag">
            <span>SPOT: {curr['spot_price']:.0f}</span>
            <span>ATM IV: {curr['m1_iv']:.2f}%</span>
            <span>STRADDLE: {curr['m1_straddle']:.0f}</span>
            <span>DTE: {ctx['dte']}</span>
            <span>NEXT IV: {curr.get('m2_iv',0):.2f}%</span>
        </div>
    """, unsafe_allow_html=True)

    base_regime = ctx['regime'].replace(" (LATE)", "")
    if base_regime in STRATEGY_MATRIX:
        strat_data = STRATEGY_MATRIX[base_regime]
        st.markdown(f"""
        <div class="strat-box" style="border-left-color: {ctx['color']};">
            <div class="strat-header" style="color: {ctx['color']}">🧭 Risk-Compatible Strategies for {ctx['regime']}</div>
            <div style="display: flex; gap: 40px;">
                <div style="flex: 1;">
                    <strong style="color: #ddd;">✅ WHAT WORKS:</strong>
                    <ul class="strat-list">
                        {''.join([f'<li>{item}</li>' for item in strat_data['works']])}
                    </ul>
                </div>
                <div style="flex: 1;">
                    <strong style="color: #ddd;">🛠️ SUGGESTED STRATEGIES:</strong>
                    <ul class="strat-list">
                        {''.join([f'<li>{item}</li>' for item in strat_data['strategies']])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 VOLSTATE Analytics</div>', unsafe_allow_html=True)
    
    # --- CHART LOGIC (FILTERED TO MAX 2 MONTHS / 60 ROWS) ---
    # We use df_selected which ends at the selected date
    df_chart = df_selected.sort_values('timestamp', ascending=True).tail(60).copy()

    df_regime = calculate_historical_regime(df_chart)
    if not df_regime.empty:
        # --- CHANGED TO SCATTER SQUARES ---
        df_regime['y_val'] = 1 # Dummy Y-axis to align squares
        fig_regime = px.scatter(df_regime, x="timestamp", y="y_val", color="regime", 
                            color_discrete_map={
                                "COMPRESSION": "#28a745", "TRANSITION": "#ffc107", 
                                "EXPANSION": "#fd7e14", "STRESS": "#dc3545"
                            },
                            symbol_sequence=['square'], # Force square shape
                            title="<b>Historical Regime Timeline (Last 60 Days)</b>")
        
        fig_regime.update_traces(marker=dict(size=15)) # Make squares big
        # Hide Y-axis and Legend
        fig_regime.update_layout(template="plotly_dark", height=130, showlegend=False, 
                                 yaxis=dict(visible=False, showgrid=False), 
                                 xaxis=dict(showgrid=False),
                                 xaxis_title=None, margin=dict(t=30, b=10))
        st.plotly_chart(fig_regime, use_container_width=True)

    df_hist = df_chart.copy()
    df_hist['slope'] = df_hist['m3_iv'] - df_hist['m1_iv']
    df_hist['slope_col'] = np.where(df_hist['slope'] >= 0, '#00cc00', '#ff0000')
    df_hist['std_pct'] = df_hist['m1_straddle'].pct_change() * 100
    df_hist['std_col'] = np.where(df_hist['std_pct'] <= 0, '#00cc00', '#ff0000') 
    df_hist['log_ret'] = np.log(df_hist['spot_price'] / df_hist['spot_price'].shift(1))
    df_hist['rv_5d'] = df_hist['log_ret'].rolling(window=5).std() * np.sqrt(252) * 100
    df_hist['vrp'] = df_hist['m1_iv'] - df_hist['rv_5d']
    df_hist['vrp_col'] = np.where(df_hist['vrp'] > 0, '#00cc00', '#ff0000') 
    daily_iv = (df_hist['m1_iv'] / 100) / np.sqrt(252)
    spot_pct = df_hist['spot_price'].pct_change()
    df_hist['sd_move'] = (spot_pct / daily_iv.shift(1)).abs().fillna(0)

    fig_spot = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spot.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['spot_price'], line=dict(color='#3498db', width=2), name="Nifty Spot"), secondary_y=False)
    fig_spot.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['m1_straddle'], line=dict(color='#e74c3c', width=2, dash='dot'), name="ATM Straddle"), secondary_y=True)
    fig_spot.update_layout(title="<b>Nifty Spot vs ATM Straddle Price Trend</b>", template="plotly_dark", height=350, margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_spot.update_yaxes(title_text="Spot Price", secondary_y=False)
    fig_spot.update_yaxes(title_text="Straddle Premium", secondary_y=True)
    st.plotly_chart(fig_spot, use_container_width=True)

    col_r1_1, col_r1_2 = st.columns(2)
    with col_r1_1:
        fig_slope = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['slope'], marker_color=df_hist['slope_col']))
        fig_slope.update_layout(title="<b>Term Structure Slope</b><br><span style='font-size:10px; color:gray'>Green=Contango (Safe), Red=Inverted (Panic)</span>", 
                                template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False, 
                                bargap=0.1)
        st.plotly_chart(fig_slope, use_container_width=True)

    with col_r1_2:
        fig_std = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['std_pct'], marker_color=df_hist['std_col']))
        fig_std.update_layout(title="<b>Daily Straddle Change %</b><br><span style='font-size:10px; color:gray'>Green=Decay (Profit), Red=Expansion (Loss)</span>", 
                              template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
                              bargap=0.1)
        st.plotly_chart(fig_std, use_container_width=True)

    col_r2_1, col_r2_2 = st.columns(2)
    with col_r2_1:
        fig_vrp = go.Figure(go.Bar(x=df_hist['timestamp'], y=df_hist['vrp'], marker_color=df_hist['vrp_col']))
        fig_vrp.update_layout(title="<b>VRP Index (IV - RV) [Edge]</b><br><span style='font-size:10px; color:gray'>Red = Negative Edge (No Trade Zone)</span>", 
                              template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
                              bargap=0.1)
        st.plotly_chart(fig_vrp, use_container_width=True)
        
    with col_r2_2:
        fig_skew = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['skew_index'], mode='lines', line=dict(color='#3498db', width=2), fill='tozeroy'))
        fig_skew.update_layout(title="<b>Skew Index</b><br><span style='font-size:10px; color:gray'>Rising = Hedging Demand (Warning)</span>", 
                               template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_skew, use_container_width=True)

    col_r3_1, col_r3_2 = st.columns(2)
    with col_r3_1:
        fig_vvix = go.Figure(go.Scatter(x=df_hist['timestamp'], y=df_hist['vvix'], mode='lines', line=dict(color='#f1c40f', width=2)))
        fig_vvix.update_layout(title="<b>INDIA VIX</b><br><span style='font-size:10px; color:gray'>Market Fear Gauge</span>", 
                               template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vvix, use_container_width=True)
        
    with col_r3_2:
        fig_sd = go.Figure()
        fig_sd.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['sd_move'], fill='tozeroy', mode='lines', line=dict(color='#9b59b6')))
        fig_sd.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig_sd.update_layout(title="<b>Price Displacement (SD)</b><br><span style='font-size:10px; color:gray'>Spikes > 1.0 indicate Dislocation</span>", 
                             template="plotly_dark", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_sd, use_container_width=True)

    # --- RAW DATA SECTION ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    # Label identifying the start date
    start_date_label = df_all['timestamp'].min().strftime('%d %b %Y')
    with st.expander(f"📂 View Raw Database (Data from {start_date_label})"):
        st.dataframe(df_all.style.format("{:.2f}", subset=['spot_price', 'm1_straddle', 'm1_iv', 'm2_iv', 'm3_iv', 'skew_index', 'vvix']))

    # --- USER GUIDE SECTION ---
    with st.expander("📚 How to Read This Dashboard (User Guide)"):
        st.markdown("""
        ### **1. The Workflow**
        1.  **Check the Regime Label:** Top center. Is it Green (Safe), Yellow (Caution), or Red (Danger)?
        2.  **Check Confidence:** High confidence means 0-1 signals (Compression) or 3+ signals (Expansion). Low confidence means the market is confused.
        3.  **Inspect the Grid:** Which specific tiles are lit up? (e.g., if only 'Skew' is lit, big players are hedging).
        4.  **Select Strategy:** Look at the "Suggested Strategies" box for trades that match the current math.

        ### **2. The 6 Logic Gates (Inputs & Meaning)**
        | Signal | Name | Logic | Interpretation |
        | :--- | :--- | :--- | :--- |
        | **T1** | **ATM IV Rising** | IV Rising 2 days in a row? | **Vol Awakening.** Options are getting more expensive. |
        | **T2** | **Straddle Stalled** | Straddle Price drop < 0.2%? | **Theta Failure.** Premium is NOT decaying. Buying pressure > Time decay. |
        | **T3** | **Back Month Bid** | Far Month IV > Near Month IV? | **Future Fear.** Traders expect volatility to persist. |
        | **T4** | **Term Structure** | Near IV > Far IV? (Inversion) | **Panic.** Immediate demand for protection is extreme. |
        | **T5** | **Tail Risk** | Skew Index Rising? | **Crash Hedge.** Puts are becoming expensive relative to Calls. |
        | **T6** | **Disconnect** | Spot UP + IV UP? | **Instability.** Rally is fueled by fear/FOMO, not stability. |

        ### **3. The Regimes (Results)**
        * **COMPRESSION (Green):** Market is asleep. **Strategy:** Harvest Theta (Iron Condors, Short Straddles).
        * **TRANSITION (Yellow):** Waking up. **Strategy:** Reduce size, move to Defined Risk (Spreads).
        * **EXPANSION (Orange):** Trending/Exploding. **Strategy:** Long Volatility (Debit Spreads), Trend Following.
        * **STRESS (Red):** Panic/Crash. **Strategy:** Cash or Hedges only. Do not sell premium.
        """)

if __name__ == "__main__":
    main()
