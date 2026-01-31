import streamlit as st

def render_documentation_tab():
    # --- CSS: CLEAN CENTERED LAYOUT ---
    st.markdown("""
        <style>
        /* Center content and limit width for readability */
        .docs-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Executive Summary Box - Fixed Layout */
        .exec-box {
            background-color: #0d1117;
            border-left: 5px solid #ffc107;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 30px;
            color: #c9d1d9;
            font-size: 16px;
            line-height: 1.6;
        }

        /* Table of Contents Styling */
        .toc-list {
            list-style: none;
            padding: 0;
        }
        .toc-item {
            margin-bottom: 20px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 15px;
        }
        .toc-link {
            font-size: 18px;
            font-weight: bold;
            color: #58a6ff;
            text-decoration: none;
            display: block;
            margin-bottom: 5px;
        }
        .toc-link:hover {
            color: #ffc107;
            text-decoration: underline;
        }
        .toc-desc {
            font-size: 14px;
            color: #8b949e;
            margin-left: 5px;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- WRAPPER DIV ---
    st.markdown('<div class="docs-container">', unsafe_allow_html=True)

    # --- 1. FIXED EXECUTIVE SUMMARY ---
    st.markdown("""
    # ⚡ VOLSTATE System Documentation
    
    <div class="exec-box">
        <p style="margin: 0;">
            <b>VOLSTATE</b> is a "permission engine" for short-volatility trading. It does not predict where the market is going. 
            Instead, it strictly tells you <b>when it is safe to carry risk</b>. It monitors hidden stress signals (like panic hedging and curve inversions) 
            to prevent the #1 cause of trading losses: overstaying your welcome during a market crash.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- 2. DUMBED-DOWN TABLE OF CONTENTS ---
    # Using HTML to ensure the specific layout of Link + Description
    st.markdown("""
    <h3 style="border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 20px;">📑 Table of Contents</h3>
    
    <ul class="toc-list">
        <li class="toc-item">
            <a href="#0-system-overview" class="toc-link">0. System Overview</a>
            <div class="toc-desc">The high-level logic. No math, just the "what" and "why" of the system.</div>
        </li>
        <li class="toc-item">
            <a href="#1-regime-probability-vector-rpv" class="toc-link">1. Regime Probability Vector (RPV)</a>
            <div class="toc-desc">The market weather report. It tells you if the environment is sunny (Compression), cloudy (Transition), or stormy (Stress).</div>
        </li>
        <li class="toc-item">
            <a href="#2-regime-dynamics" class="toc-link">2. Regime Dynamics</a>
            <div class="toc-desc">How fast the weather is changing. Is a storm approaching quickly, or is the sun coming out?</div>
        </li>
        <li class="toc-item">
            <a href="#3-carry-integrity-score-cis" class="toc-link">3. Carry Integrity Score (CIS)</a>
            <div class="toc-desc">The final Green/Red light. A single score that answers: "Am I allowed to trade right now?"</div>
        </li>
        <li class="toc-item">
            <a href="#4-authority-flow-critical" class="toc-link">4. Authority Flow</a>
            <div class="toc-desc">The chain of command. The math (CIS) always overrules your gut feeling.</div>
        </li>
    </ul>
    """, unsafe_allow_html=True)

    # --- 3. MAIN DOCUMENTATION BODY ---
    st.markdown(r"""
    <hr style="margin-top: 40px; margin-bottom: 40px; border-color: #444;">

    <a id="0-system-overview"></a>
    # 0. SYSTEM OVERVIEW

    <a id="01-what-volstate-is"></a>
    ### 0.1 What VOLSTATE Is
    VOLSTATE is a regime-aware risk control and discipline enforcement system designed specifically for:
    * Short volatility strategies
    * Volatility Risk Premium (VRP) harvesting
    * Monthly option cycles
    * 45 → 21 DTE holding windows

    **VOLSTATE does not forecast markets.**
    VOLSTATE governs when volatility carry is structurally permitted.

    <a id="03-architectural-layers"></a>
    ### 0.3 Architectural Layers

    | Layer | Purpose |
    | :--- | :--- |
    | **Raw Market Data** | Spot, IVs, Straddles, Skew |
    | **Likelihood Sensors** | Raw data → fuzzy structural signals |
    | **RPV Engine** | Diagnose market regime mixture |
    | **Regime Dynamics** | Momentum & acceleration |
    | **CIS Engine** | Compress diagnostics into action |
    | **Theta Cycle Controller** | Enforces SOP |

    ---

    <a id="1-regime-probability-vector-rpv"></a>
    # 1. REGIME PROBABILITY VECTOR (RPV)

    ### 1.1 Definition
    The Regime Probability Vector (RPV) represents a probability mixture of four volatility regimes:

    $$
    RPV = \{P_C, P_T, P_E, P_S\}, \quad \sum_i P_i = 1
    $$

    <a id="14-likelihood-sensors-structural-core"></a>
    ### 1.4 Likelihood Sensors (Structural Core)
    Each sensor emits regime likelihoods:
    $$
    L_{i,r} \in [0,1]
    $$
    These likelihoods are **Deterministic, Piecewise-linear, Clipped, and Non-Bayesian**. This design avoids distributional assumptions and overfitting.

    #### S1 — ATM IV Momentum
    **Input:**
    $$
    \Delta IV_1 = IV_{1,t} - IV_{1,t-1}
    $$

    **Likelihood Mapping:**
    $$
    \begin{aligned}
    L_{IV,C} &= \max\left(0, 1 - \frac{\Delta IV_1}{0.5}\right) \\
    L_{IV,T} &= \text{clip}\left(\frac{\Delta IV_1}{0.6}, 0, 1\right) \\
    L_{IV,E} &= \text{clip}\left(\frac{\Delta IV_1}{1.0}, 0, 1\right) \\
    L_{IV,S} &= \text{clip}\left(\frac{\Delta IV_1 - 0.8}{1.2}, 0, 1\right)
    \end{aligned}
    $$

    **Interpretation:**
    * Compression collapses quickly as IV rises
    * Stress activates only after large IV shocks
    * Expansion tolerates moderate IV increases

    #### S2 — Straddle Decay (Theta Efficiency)
    **Input:**
    $$
    \Delta STR\% = \frac{STR_t - STR_{t-1}}{STR_{t-1}} \times 100
    $$

    **Likelihood Mapping:**
    $$
    \begin{aligned}
    L_{STR,C} &= \begin{cases} 1 & \text{if } \Delta STR\% < -0.2 \\ 0.2 & \text{otherwise} \end{cases} \\
    L_{STR,T} &= \text{clip}\left(\frac{\Delta STR\% + 0.2}{0.4}, 0, 1\right) \\
    L_{STR,E} &= \text{clip}\left(\frac{\Delta STR\% + 0.1}{0.6}, 0, 1\right) \\
    L_{STR,S} &= \text{clip}\left(\frac{\Delta STR\% - 0.8}{1.0}, 0, 1\right)
    \end{aligned}
    $$

    **Interpretation:**
    * Compression requires actual decay
    * Stress only triggers on straddle expansion
    * This is the closest proxy to live carry P&L

    #### S6 — Spot–Vol Disconnect
    **Condition:**
    $$
    (|\Delta S| < 0.1\%) \land (\Delta IV_1 > 0.5)
    $$

    **Likelihood Mapping:**
    $$
    \begin{aligned}
    L_{DISC,C} &= 0 \\
    L_{DISC,T} &= \begin{cases} 0.4 & \text{if disconnect} \\ 0.6 & \text{otherwise} \end{cases} \\
    L_{DISC,E} &= \begin{cases} 0.7 & \text{if disconnect} \\ 0.4 & \text{otherwise} \end{cases} \\
    L_{DISC,S} &= \begin{cases} 0.9 & \text{if disconnect} \\ 0.2 & \text{otherwise} \end{cases}
    \end{aligned}
    $$

    **Interpretation:**
    * Penalizing, not dominant
    * Strong bias toward Stress when present

    <a id="16-rpv-weighting"></a>
    ### 1.6 RPV Weighting
    $$
    \mathbf{w} = \begin{cases} w_{IV} = 1.2 \\ w_{STR} = 1.2 \\ w_{BM} = 1.0 \\ w_{TS} = 1.0 \\ w_{SK} = 1.4 \\ w_{DISC} = 0.8 \end{cases}
    $$
    *Weights favor capital survival, not early responsiveness.*

    ---

    <a id="2-regime-dynamics"></a>
    # 2. REGIME DYNAMICS

    ### 2.1 RPV Drift
    $$
    Drift_r = RPV_{r,t} - RPV_{r,t-k}
    $$
    *Measures velocity, not level. Positive values indicate increasing probability.*

    ### 2.2 Pre-Stress Detector
    Triggered when **all** hold:
    1.  Stress > 20%
    2.  Stress slope > 0.08
    3.  Stress accelerating
    4.  Stress > 0.6 × Expansion

    *Early crash sensor.*

    ---

    <a id="3-carry-integrity-score-cis"></a>
    # 3. CARRY INTEGRITY SCORE (CIS)

    ### 3.1 Definition
    $$
    CIS \in [-1, +1]
    $$
    **Answers:** *Am I allowed to carry short gamma right now?*

    ### 3.2 CIS Inputs

    **Structural**
    * $C$: Compression probability
    * $T$: Transition probability
    * $S$: Stress probability

    **Performance**
    $$
    \hat{\theta} = \text{clip}\left(\frac{-\Delta STR\%}{0.25}, -1, 1\right)
    $$

    $$
    \hat{K} = \text{clip}\left(\frac{IV_2 - IV_1}{1.0}, -1, 1\right)
    $$

    **Instability**
    $$
    \hat{D} = \text{clip}\left(\frac{\Delta Stress}{0.10}, 0, 1\right)
    $$

    $$
    A = \begin{cases} 1 & \text{if accelerating} \\ 0 & \text{otherwise} \end{cases}
    $$

    <a id="33-cis-formula-corrected"></a>
    ### 3.3 CIS Formula (Corrected)
    $$
    \boxed{
    CIS = + 0.40(C + T) + 0.20\hat{\theta} + 0.15\hat{K} - 0.50S - 0.30\hat{D} - 0.20A
    }
    $$
    *Clipped to $[-1, +1]$.*

    ### 3.4 CIS Interpretation

    | CIS | Action |
    | :--- | :--- |
    | **> +0.35** | Full carry |
    | **+0.15 to +0.35** | Reduce size |
    | **0 to +0.15** | No new risk |
    | **−0.25 to 0** | Exit bias |
    | **< −0.25** | Immediate exit |

    ---

    <a id="4-authority-flow-critical"></a>
    # 4. AUTHORITY FLOW (CRITICAL)

    $$
    \boxed{ RPV \rightarrow CIS \rightarrow ACTION }
    $$

    1.  **RPV** describes structure.
    2.  **CIS** governs permission.
    3.  **Actions** follow SOP.

    ## 5. FINAL PHILOSOPHY
    VOLSTATE does not predict volatility.
    It enforces discipline on those who profit from selling it.
    """, unsafe_allow_html=True)

    # --- CLOSE WRAPPER DIV ---
    st.markdown('</div>', unsafe_allow_html=True)