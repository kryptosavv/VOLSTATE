import streamlit as st

def render_documentation_tab():
    # --- CSS: CLEAN CENTERED LAYOUT ---
    st.markdown("""
    <style>
    .docs-container {
        max-width: 860px;
        margin: 0 auto;
        padding: 20px;
        color: #c9d1d9; /* Default text color */
    }

    /* Fixed Executive Box Styling */
    .doc-exec-box {
        background-color: #0d1117;
        border-left: 5px solid #ffc107;
        padding: 25px;
        border-radius: 4px;
        margin-bottom: 30px;
        font-size: 16px;
        line-height: 1.6;
        display: flex;
        flex-direction: column;
    }

    .doc-exec-title {
        color: #ffc107;
        font-weight: 900;
        font-size: 20px;
        margin-bottom: 15px;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        border-bottom: 1px solid rgba(255, 193, 7, 0.2);
        padding-bottom: 10px;
        display: block;
    }

    /* Headers */
    h2 {
        margin-top: 60px !important;
        margin-bottom: 20px !important;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
        color: #58a6ff;
    }

    h3 {
        margin-top: 30px !important;
        color: #c9d1d9;
    }
    /* Standard Table Styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    th {
        background-color: #161b22;
        color: #ffc107;
        font-weight: bold;
        text-align: left;
        padding: 10px;
        border-bottom: 2px solid #30363d;
    }
    td {
        padding: 10px;
        border-bottom: 1px solid #30363d;
        vertical-align: top;
        line-height: 1.6;
    }
    tr:hover {background-color: #0d1117;}

    /* TOC Styling */
    .toc-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 40px;
    }
    .toc-header {
        font-size: 18px;
        font-weight: bold;
        color: #c9d1d9;
        margin-bottom: 10px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 5px;
    }
    .toc-list {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }
    .toc-item {
        margin: 8px 0;
    }
    .toc-link {
        color: #58a6ff;
        text-decoration: none;
        font-family: 'Segoe UI', monospace;
        font-size: 15px;
    }
    .toc-link:hover {
        text-decoration: underline;
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- WRAPPER DIV ---
    st.markdown('<div class="docs-container">', unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown("#  🔐  VOLSTATE SYSTEM — TECHNICAL DOCUMENTATION")

    # --- EXECUTIVE SUMMARY ---
    st.markdown("""
    <div class="doc-exec-box">
        <div class="doc-exec-title">EXECUTIVE SUMMARY</div>
        <p>
            VOLSTATE is a deterministic, regime-aware permission system for volatility trading.
            It does not predict markets, forecast volatility, or optimize returns. Its sole purpose is to
            decide when it is structurally safe to carry risk and when convexity is justified.
        </p>
        <p>
            The system converts raw volatility structure into explicit regime probabilities, tracks how those
            regimes evolve over time, and enforces discipline through two permission scores: the
            <strong>Carry Integrity Score (CIS)</strong> and the <strong>Convexity Permission Score (CPS)</strong>.
            All trading decisions must flow from these permissions. VOLSTATE is intentionally conservative and
            biased toward early exits; it exists to prevent catastrophic losses caused by overstaying in hostile
            volatility regimes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- TABLE OF CONTENTS ---
    st.markdown("""
    <div class="toc-box">
        <div class="toc-header"> 📑  INDEX</div>
        <ul class="toc-list">
            <li class="toc-item"><a class="toc-link" href="#0-system-architecture">0. System Architecture</a></li>
            <li class="toc-item"><a class="toc-link" href="#1-raw-input-space-canonical">1. Raw Input Space (Canonical)</a></li>
            <li class="toc-item"><a class="toc-link" href="#2-regime-probability-vector-rpv">2. Regime Probability Vector (RPV)</a></li>
            <li class="toc-item"><a class="toc-link" href="#3-regime-dynamics-momentum-layer">3. Regime Dynamics (Momentum Layer)</a></li>
            <li class="toc-item"><a class="toc-link" href="#4-carry-integrity-score-cis">4. Carry Integrity Score (CIS)</a></li>
            <li class="toc-item"><a class="toc-link" href="#4-1-why-cis-is-intentionally-conservative">4.1 Why CIS Is Intentionally Conservative</a></li>
            <li class="toc-item"><a class="toc-link" href="#5-cis-context-interpretability-layer">5. CIS Context (Interpretability Layer)</a></li>
            <li class="toc-item"><a class="toc-link" href="#6-convexity-permission-score-cps">6. Convexity Permission Score (CPS)</a></li>
            <li class="toc-item"><a class="toc-link" href="#7-action-matrix">7. Action Matrix</a></li>
            <li class="toc-item"><a class="toc-link" href="#8-how-regimes-are-decided-summary">8. How Regimes Are Decided (Summary)</a></li>
            <li class="toc-item"><a class="toc-link" href="#9-failure-modes">9. Failure Modes</a></li>
            <li class="toc-item"><a class="toc-link" href="#10-fail-safe-overrides-non-negotiable">10. Fail-Safe Overrides (Non-Negotiable)</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 0. SYSTEM ARCHITECTURE
    # =========================
    st.markdown("## 0. SYSTEM ARCHITECTURE")
    st.markdown("""
    VOLSTATE operates in a strict, non-overridable hierarchy. No layer may bypass or override the one above it.
    """)
    st.latex(r"""
    \text{Raw Inputs} \rightarrow \text{Sensors} \rightarrow \text{Likelihoods} \rightarrow \text{RPV} \rightarrow \text{Dynamics} \rightarrow (\text{CIS}, \text{CPS}) \rightarrow \text{Trades}
    """)

    # =========================
    # 1. RAW INPUT SPACE
    # =========================
    st.markdown("## 1. RAW INPUT SPACE (CANONICAL)")
    st.markdown("All calculations are front-month anchored, regardless of which expiry is traded.")
    st.markdown("""
    | Variable | Symbol | Meaning |
    | :--- | :---: | :--- |
    | Spot Price | $S_t$ | Index level |
    | M1 ATM IV | $IV_1$ | Front-month implied volatility |
    | M2 ATM IV | $IV_2$ | Next-month implied volatility |
    | M3 ATM IV | $IV_3$ | Far-month implied volatility |
    | ATM Straddle | $STR_t$ | Price of ATM straddle |
    | Skew Index | $SK_t$ | Downside tail demand |
    """)
    st.markdown("### Derived Quantities")
    st.latex(r"""
    \begin{aligned}
    \Delta IV_1 &= IV_{1,t} - IV_{1,t-1} \\
    \Delta STR\% &= \frac{STR_t - STR_{t-1}}{STR_{t-1}} \times 100 \\
    \Delta SK &= SK_t - SK_{t-1}
    \end{aligned}
    """)

    # =========================
    # 2. RPV
    # =========================
    st.markdown("## 2. REGIME PROBABILITY VECTOR (RPV)")
    st.markdown("### Purpose")
    st.markdown("Diagnose the current volatility regime mixture (structure only). Markets are never assumed to be in a single regime.")
    st.latex(r"""
    RPV = \{P_C, P_T, P_E, P_S\}, \quad \sum_r P_r = 1
    """)
    st.markdown("""
    | Regime | Symbol | Meaning |
    | :--- | :---: | :--- |
    | **Compression** | C | Decay dominates |
    | **Transition** | T | Structure weakening |
    | **Expansion** | E | Directional volatility |
    | **Stress** | S | Panic / tail risk |
    """)
    st.markdown("### A. RPV Inputs (Exhaustive)")

    st.markdown("""
    <table>
    <thead>
    <tr>
    <th>Signal</th>
    <th>Description</th>
    <th>Formula</th>
    <th>Weight</th>
    <th>Why it matters</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>ATM IV Momentum</strong></td>
    <td>Forward vol repricing</td>
    <td>ΔIV<sub>1</sub></td>
    <td style="text-align:center">1.2</td>
    <td>IV reprices before spot</td>
    </tr>
    <tr>
    <td><strong>Straddle % Change</strong></td>
    <td>Theta efficiency</td>
    <td>ΔSTR%</td>
    <td style="text-align:center">1.2</td>
    <td>Closest proxy to real carry P&L</td>
    </tr>
    <tr>
    <td><strong>Back-Month Spread</strong></td>
    <td>Fear propagation</td>
    <td>(IV<sub>2</sub> − IV<sub>2,−1</sub>) − ΔIV<sub>1</sub></td>
    <td style="text-align:center">1.0</td>
    <td>Distinguishes local vs systemic</td>
    </tr>
    <tr>
    <td><strong>Term Structure</strong></td>
    <td>Systemic stress</td>
    <td>IV<sub>3</sub> − IV<sub>1</sub></td>
    <td style="text-align:center">1.0</td>
    <td>Inversion = capital flight</td>
    </tr>
    <tr>
    <td><strong>Skew Change</strong></td>
    <td>Tail hedging</td>
    <td>ΔSK</td>
    <td style="text-align:center">1.4</td>
    <td>Smart money panic signal</td>
    </tr>
    <tr>
    <td><strong>Spot–Vol Disconnect</strong></td>
    <td>Hidden panic</td>
    <td>|ΔS<sub>t</sub>| < ε  AND  ΔIV<sub>1</sub> > θ</td>
    <td style="text-align:center">0.8</td>
    <td>Non-price fear</td>
    </tr>
    </tbody>
    </table>
    <div style="font-size: 12px; color: #888; font-style: italic;">*Design note: Weights are intentionally asymmetric.
    Signals closest to dealer P&L reality (straddle, skew) dominate.*</div>
    <br>
    """, unsafe_allow_html=True)
    st.markdown("### B. Signal → Regime Probability Mapping")
    st.markdown("Each signal $i$ emits regime likelihoods:")
    st.latex(r"L_{i,r} \in [0,1]")

    st.markdown("Weighted aggregation:")
    st.latex(r"Score_r = \sum_i w_i \cdot L_{i,r}")

    st.markdown("Normalization:")
    st.latex(r"RPV_r = \frac{Score_r}{\sum_r Score_r}")
    st.markdown("### C. Explicit Regime Logic")
    st.markdown("""
    | Regime | Structural Signature |
    | :--- | :--- |
    | **Compression** | IV flat/down + straddles decaying + positive slope |
    | **Transition** | Mixed signals + early IV upticks |
    | **Expansion** | IV rising + straddles expanding |
    | **Stress** | Front IV spike + straddle explosion + inversion |
    """)
    st.info("**Important:** RPV never forces a single regime. Mixtures are intentional.")

    # =========================
    # 3. REGIME DYNAMICS
    # =========================
    st.markdown("## 3. REGIME DYNAMICS (MOMENTUM LAYER)")
    st.markdown("### Purpose")
    st.markdown("Detect where regimes are *heading*, not just where they are.")
    st.markdown("""
    | Input | Formula | Why |
    | :--- | :--- | :--- |
    | **RPV Drift** | $RPV_r(t) - RPV_r(t-k)$ | Velocity of change |
    | **Stress Acceleration** | $\Delta S_t > \Delta S_{t-1}$ | Convex loss risk |
    | **Expansion Comparison** | $S > 0.6E$ | Disorder dominance |
    """)

    st.markdown("### Stress Drift Persistence Filter")
    st.markdown("""
    Stress drift is considered actionable only if it persists for **two consecutive windows**. This prevents CIS degradation from single-day noise or expiry artifacts.
    <br><em>Rationale: This converts drift from a sensitivity signal into a structural signal.</em>
    """, unsafe_allow_html=True)

    st.markdown("### Pre-Stress Trigger")
    st.markdown("Triggers if **ALL** of the following hold:")
    st.markdown("""
    1.  Stress > 20%
    2.  Stress drift > 0.08
    3.  Stress accelerating
    4.  Stress > 0.6 × Expansion
    """)
    st.warning("Pre-Stress triggers escalate exit urgency and contextual warnings but **do not directly reduce CIS** unless stress thresholds are breached.")

    # =========================
    # 4. CARRY INTEGRITY SCORE (CIS)
    # =========================
    st.markdown("## 4. CARRY INTEGRITY SCORE (CIS)")
    st.markdown("### Purpose")
    st.markdown("Binary question: **Is short gamma structurally allowed?** CIS is a gated score, not a continuous penalty model. Stress penalties only activate beyond a structural threshold.")

    st.markdown("### CIS Definition (Canonical)")
    st.markdown("If $S < 0.18$:")
    st.latex(r"CIS = 0.40(C+T) + 0.20\hat{\theta} + \hat{K}")
    
    st.markdown("If $S \ge 0.18$:")
    st.latex(r"""
    CIS = 0.40(C+T) + 0.20\hat{\theta} + \hat{K} - 0.30S - 0.20\text{clip}(\Delta S / 0.20) - 0.10A
    """)
    st.caption("No stress penalty below 18%. Drift is persistence-gated. Acceleration is binary.")

    st.markdown("### CIS Inputs Breakdown")
    st.markdown("""
    | Component | Formula | Weight | Why |
    | :--- | :--- | :---: | :--- |
    | **Structural Carry** | $C + T$ | +0.40 | Carry survives only in calm regimes |
    | **Theta Efficiency** | $\hat{\\theta} = \text{clip}(-\Delta STR\%/0.30, -0.8, +1.0)$ | +0.20 | Asymmetric saturation; avoids over-penalizing large decay |
    | **Carry Insulation** | Deadbanded (see below) | Variable | Back-month buffer is binary-defensive |
    | **Stress Level** | $-S$ | -0.30 (Gated) | Convex losses dominate |
    | **Stress Drift** | $-\text{clip}(\Delta S/0.20)$ | -0.20 (Gated) | Prevents overstaying |
    | **Stress Accel** | $-A$ | -0.10 (Gated) | Kill-switch bias |
    """)

    st.markdown("### Carry Insulation Logic")
    st.markdown("Carry insulation is **binary-defensive**, not yield-seeking.")
    st.markdown("""
    | Condition | CIS Effect |
    | :--- | :--- |
    | $IV_2 - IV_1 < -0.25$ | **-0.15** (Penalty) |
    | $-0.25 \le \text{Spread} \le 0.25$ | **0** (Deadband) |
    | $IV_2 - IV_1 > +0.25$ | **+0.10** (Bonus) |
    """)

    # =========================
    # 4.1 WHY CIS IS CONSERVATIVE
    # =========================
    st.markdown("## 4.1 WHY CIS IS INTENTIONALLY CONSERVATIVE")
    st.markdown("""
    * CIS is **permission**, not edge.
    * CIS decays faster than P&L (leading indicator).
    * False negatives (missing a trade) are preferred over false positives (taking a bad trade).
    * CIS is designed to **exit early, not optimally**.
    """)

    st.markdown("### CIS Permission Bands")
    st.markdown("""
    | CIS | Meaning |
    | :--- | :--- |
    | **> 0.35** | Full carry |
    | **0.15 → 0.35** | Controlled carry |
    | **-0.05 → 0.15** | Tolerance only |
    | **< -0.05** | No carry |
    """)
    st.markdown("*Flat CIS ≠ inactivity. It means maintenance only, no expansion.*")

    # =========================
    # 5. CIS CONTEXT
    # =========================
    st.markdown("## 5. CIS CONTEXT (INTERPRETABILITY LAYER)")
    st.markdown("### Purpose")
    st.markdown("Explain *why* CIS is low or high. This prevents false panic exits.")
    st.markdown("""
    | Priority | Condition | Context Label |
    | :---: | :--- | :--- |
    | 1 | Stress > 20% | Rising stress probability |
    | 2 | Effective Drift > 0 | Accelerating stress drift |
    | 3 | Straddle decay stalled | Theta breakdown |
    | 4 | IV repricing | Front-month repricing |
    | 5 | Term erosion | Curve deterioration |
    | 6 | Else | Carry structure stable |
    """)

    # =========================
    # 6. CONVEXITY PERMISSION SCORE (CPS)
    # =========================
    st.markdown("## 6. CONVEXITY PERMISSION SCORE (CPS)")
    st.markdown("### Purpose")
    st.markdown("Decide when owning gamma is justified.")
    st.markdown("""
    | Input | Formula | Weight | Why |
    | :--- | :--- | :---: | :--- |
    | **Expansion + Stress** | $E + S$ | 0.40 | Convexity needs disorder |
    | **Gamma Effectiveness** | $\text{clip}(\Delta STR\%/0.30)$ | 0.25 | Must actually pay |
    | **Term Inversion** | $-(IV_3 - IV_1)$ | 0.15 | Crash confirmation |
    | **Lag Convexity** | $IV_2 - IV_1$ | 0.10 | Delayed repricing |
    | **Skew Accel** | Boolean | 0.10 | Tail demand |
    """)
    st.caption("CPS is capped to avoid overpaying for convexity.")

    # =========================
    # 7. ACTION MATRIX
    # =========================
    st.markdown("## 7. ACTION MATRIX")
    st.markdown("""
    | CIS | CPS | Action |
    | :---: | :---: | :--- |
    | **+** | **-** | Carry primary |
    | **+** | **+** | Carry + convex overlay |
    | **-** | **+** | Convexity primary |
    | **-** | **-** | Flat / hedge only |
    """)

    # =========================
    # 8. SUMMARY
    # =========================
    st.markdown("## 8. HOW REGIMES ARE DECIDED (SUMMARY)")
    st.markdown("""
    * **RPV** defines regime
    * **Dynamics** define trajectory
    * **CIS/CPS** define permission
    """)
    st.markdown("""
    | Situation | Interpretation |
    | :--- | :--- |
    | Compression high, Stress flat | Stable carry |
    | Transition rising, Stress drifting | Early danger |
    | Expansion dominant | Convexity viable |
    | Stress accelerating | Immediate exit |
    """)

    # =========================
    # 9. FAILURE MODES
    # =========================
    st.markdown("## 9. FAILURE MODES")
    st.markdown("Explicit acknowledgement of where the system may lag or fail, and the accepted design trade-offs.")
    
    st.markdown("""
    <table>
    <thead>
    <tr>
    <th>Failure Mode</th>
    <th>Description</th>
    <th>Potential Impact</th>
    <th>Mitigation Controls</th>
    <th>Design Acceptance</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>Lag in Ultra-Fast Shock Events</strong></td>
    <td>System operates on end-of-interval volatility structure. Sudden intraday or overnight gaps may occur before signals update.</td>
    <td>CIS may remain permissive briefly before shock. Initial loss may occur prior to enforced exit.</td>
    <td>• Event-based overrides<br>• Gap-size fail-safes<br>• Conservative sizing by design</td>
    <td>Intentional trade-off. Exiting before structure changes would require prediction, which is explicitly avoided.</td>
    </tr>
    <tr>
    <td><strong>False Negatives in Low-Volatility Grind</strong></td>
    <td>Prolonged suppressed volatility with mild deterioration. Structural decay may be slow and non-disruptive.</td>
    <td>Lower realized Sharpe. Delayed exits from gradually degrading carry.</td>
    <td>• Stress drift penalties<br>• CIS temporal persistence rules</td>
    <td>Survivability prioritized over smooth or optimal returns.</td>
    </tr>
    <tr>
    <td><strong>False Positives from Macro Hedging</strong></td>
    <td>Temporary IV / skew elevation from hedging flows. No follow-through into realized volatility.</td>
    <td>Premature CIS decline. Early exit from otherwise viable carry.</td>
    <td>• Multi-sensor confirmation requirement<br>• Regime mixture representation instead of binary states</td>
    <td>Early exit preferred to overstaying during latent stress.</td>
    </tr>
    <tr>
    <td><strong>No Directional Alpha Awareness</strong></td>
    <td>No price forecasts, trends, or directional indicators used. Volatility structure only.</td>
    <td>Missed opportunities where direction is clear but vol is neutral.</td>
    <td>None (explicitly out of scope).</td>
    <td>Deliberate exclusion. System governs risk, not return maximization.</td>
    </tr>
    <tr>
    <td><strong>Dependence on Volatility Data Quality</strong></td>
    <td>Relies on IV, straddle, and skew aggregates. Assumes consistent and reliable market data.</td>
    <td>Temporary distortion in regime probabilities. Incorrect CIS / CPS readings.</td>
    <td>• Cross-sensor redundancy<br>• Weighted aggregation<br>• CIS/CPS conflict warnings</td>
    <td>Data integrity is a prerequisite for all quantitative systems.</td>
    </tr>
    <tr>
    <td><strong>Compression Misclassification under Suppressed Volatility</strong></td>
    <td>Heavy volatility selling or structural overwriting. Artificially stable surfaces mask underlying risk.</td>
    <td>Carry permitted longer than economically ideal.</td>
    <td>• Stress drift monitoring<br>• Back-month propagation checks</td>
    <td>Accepted trade-off in artificially controlled markets.</td>
    </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

    # =========================
    # 10. FAIL-SAFE OVERRIDES
    # =========================
    st.markdown("## 10. FAIL-SAFE OVERRIDES (NON-NEGOTIABLE)")

    st.markdown("### A. Structural Overrides")
    st.markdown("""
    * **Stress ≥ 35%:** Force no carry.
    * **M2 < M1 + IV spike:** Cut size 50%.
    * **Pre-Stress Trigger:** Exit carry.
    """)
    st.markdown("### B. Event Overrides")
    st.markdown("""
    * **RBI / Budget / CPI:** Freeze entries.
    * **Overnight gap > 1.5σ:** Reduce exposure.
    """)
    st.markdown("### C. Temporal Overrides")
    st.markdown("""
    * **CIS < 0 for 2 days:** Exit.
    * **CIS oscillates ±0.1 for 5 days:** Reduce size.
    """)
    st.markdown("### D. Human Override Rule")
    st.warning("CIS and CPS override conviction, backtests, and narratives.")

    # --- FINAL STATEMENT ---
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #30363d; color: #8b949e;">
    <em>"VOLSTATE is not designed to make trading exciting. It is designed to make survival inevitable."</em>
    </div>
    """, unsafe_allow_html=True)

    # --- END WRAPPER ---
    st.markdown('</div>', unsafe_allow_html=True)