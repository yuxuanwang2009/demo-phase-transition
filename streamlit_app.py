"""
Van der Waals Equation & Mean-Field Theory — Streamlit Web App
===============================================================

Interactive visualization with Plotly animation sliders.
Run with: streamlit run streamlit_app.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Van der Waals Equation & Mean-Field Theory",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Model parameters ────────────────────────────────────────────────────
Omega = 0.02  # hard-core volume (Ω)
U0 = 1.5      # interaction strength
u = Omega * U0
a = u / 2.0   # VdW attraction
b = Omega / 2.0  # VdW excluded volume

# Critical point: β_c = 27b/(8a), v_c = 3b, P_c = a/(27b²)
beta_c = 27 * b / (8 * a)
v_c = 3 * b
P_c = a / (27 * b**2)

# ── Plotting ranges ─────────────────────────────────────────────────────
v_min = b + 1e-6
v_max = 0.4
P_lo, P_hi = 0.2, 10.0

# ── Colors ──────────────────────────────────────────────────────────────
C_STABLE = "#16a34a"    # green
C_METASTABLE = "#f59e0b"  # amber
C_UNSTABLE = "#dc2626"   # red
C_ISOTHERM = "#2563eb"   # blue

LABEL_COLOR = {
    "stable": C_STABLE,
    "metastable": C_METASTABLE,
    "unstable": C_UNSTABLE,
}

# ── Physics helpers ─────────────────────────────────────────────────────
def neg_beta_G(v, beta, pressure):
    """−βG(v) = β p v − β u/(2v) − ln(v − b)"""
    return beta * pressure * v - beta * u / (2.0 * v) - np.log(v - b)

def d2_neg_beta_G(v, beta):
    """d²(−βG)/dv² = −β u / v³ + 1/(v−b)²"""
    return -beta * u / v**3 + 1.0 / (v - b)**2

def vdw_pressure(v, beta):
    """P(v) from VdW equation: P = 1/(β(v−b)) − a/v²"""
    return 1.0 / (beta * (v - b)) - a / v**2

def find_roots(beta, pressure):
    """Find volumes where the VdW isotherm crosses P = pressure."""
    v_fine = np.linspace(v_min, v_max, 4000)
    P_curve = vdw_pressure(v_fine, beta)
    diff = P_curve - pressure
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    roots = []
    for idx in sign_changes:
        v0, v1 = v_fine[idx], v_fine[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        roots.append(v0 - d0 * (v1 - v0) / (d1 - d0))
    return np.array(roots)

def find_maxwell_pressure(beta):
    """Find the Maxwell construction equilibrium pressure via the equal-area rule.

    Returns (P_eq, v_liquid, v_gas) or (None, None, None) if above critical T.
    """
    v_dense = np.linspace(v_min, v_max, 5000)
    P_dense = vdw_pressure(v_dense, beta)

    # Locate spinodal points (local extrema of P(v))
    dP = np.diff(P_dense)
    extrema_idx = np.where(np.diff(np.sign(dP)))[0]

    if len(extrema_idx) < 2:
        return None, None, None

    P_at_extrema = [P_dense[idx + 1] for idx in extrema_idx]
    P_lo_loop = min(P_at_extrema)
    P_hi_loop = max(P_at_extrema)

    if P_lo_loop >= P_hi_loop:
        return None, None, None

    def area_diff(P_try):
        rts = find_roots(beta, P_try)
        if len(rts) < 3:
            return None
        v1, v3 = rts[0], rts[-1]
        v_int = np.linspace(v1, v3, 2000)
        P_int = vdw_pressure(v_int, beta)
        integrand = P_int - P_try
        return np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(v_int))

    # Bisection to find P where equal-area condition holds.
    # The lower bound must be above P(v_max) so that 3 roots exist within [v_min, v_max].
    P_right_boundary = vdw_pressure(v_max, beta)
    lo = max(P_lo_loop, P_right_boundary) + 1e-6
    hi = P_hi_loop - 1e-6
    if lo >= hi:
        return None, None, None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        ad = area_diff(mid)
        if ad is None:
            break
        if ad > 0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-12:
            break

    P_eq = (lo + hi) / 2.0
    roots_eq = find_roots(beta, P_eq)
    if len(roots_eq) >= 3:
        return P_eq, roots_eq[0], roots_eq[-1]
    return None, None, None

def classify_roots(roots, beta, pressure):
    """Classify each root as 'stable', 'metastable', or 'unstable'."""
    if len(roots) == 0:
        return []

    d2 = d2_neg_beta_G(roots, beta)
    G_vals = neg_beta_G(roots, beta, pressure)

    labels = []
    for i in range(len(roots)):
        if d2[i] > 0:
            labels.append("_stable_candidate")
        else:
            labels.append("unstable")

    candidates = [(i, G_vals[i]) for i, l in enumerate(labels)
                  if l == "_stable_candidate"]

    if len(candidates) == 1:
        labels[candidates[0][0]] = "stable"
    elif len(candidates) >= 2:
        best_idx = min(candidates, key=lambda x: x[1])[0]
        for i, _ in candidates:
            labels[i] = "stable" if i == best_idx else "metastable"

    return labels

# ── Streamlit App ───────────────────────────────────────────────────────

st.title("⚛️ Van der Waals Equation & Mean-Field Theory")
st.markdown("Interactive visualization of phase transitions, stability, and free energy landscapes.")

# Sidebar controls
with st.sidebar:
    st.header("Stability Classification")
    st.markdown(r"""
    - 🟢 **Stable**: Global minimum of $G$
    - 🟠 **Metastable**: Local minimum of $G$
    - 🔴 **Unstable**: Local maximum of $G$ (saddle)
    """)

    st.divider()

    st.markdown("### Controls")

    # Button to set critical point
    if st.button("⚡ Set to Critical Point", use_container_width=True):
        st.session_state.beta_slider = beta_c
        st.session_state.pressure_slider = P_c

    # Sliders with default values
    beta = st.slider(r"$\beta$ (inverse temperature)", min_value=1.5, max_value=4.0,
                     value=2.5, step=0.01, key='beta_slider')
    pressure = st.slider(r"$P$ (pressure)", min_value=P_lo, max_value=P_hi,
                         value=2.3, step=0.05, key='pressure_slider')

# Compute plot data based on current slider values
v_plot = np.linspace(v_min, v_max, 600)
G_data = neg_beta_G(v_plot, beta, pressure)
P_data = vdw_pressure(v_plot, beta)
P_maxwell, v_liq, v_gas = find_maxwell_pressure(beta)
roots = find_roots(beta, pressure)
labels = classify_roots(roots, beta, pressure)
G_at_roots = neg_beta_G(roots, beta, pressure) if len(roots) > 0 else []

# Create figure
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Free Energy Landscape (log scale)", "VdW Isotherm & Isobar"),
    vertical_spacing=0.15,
    row_heights=[0.5, 0.5]
)

# Free energy curve
mask = G_data > 0
fig.add_trace(go.Scatter(x=v_plot[mask], y=G_data[mask], mode='lines',
                         line=dict(color=C_ISOTHERM, width=2), showlegend=False),
              row=1, col=1)

# Free energy dots
for i in range(len(roots)):
    if i < len(G_at_roots) and G_at_roots[i] > 0:
        fig.add_trace(go.Scatter(x=[roots[i]], y=[G_at_roots[i]], mode='markers',
                                 marker=dict(size=12, color=LABEL_COLOR[labels[i]],
                                           line=dict(color='white', width=2)),
                                 showlegend=False), row=1, col=1)

# Isotherm — split into stable (solid) and unstable/metastable (dashed) via Maxwell construction
if P_maxwell is not None:
    mask_left = v_plot <= v_liq
    mask_mid = (v_plot >= v_liq) & (v_plot <= v_gas)
    mask_right = v_plot >= v_gas
    # Stable portion (liquid side)
    fig.add_trace(go.Scatter(x=v_plot[mask_left], y=P_data[mask_left], mode='lines',
                             line=dict(color=C_ISOTHERM, width=2), showlegend=False),
                  row=2, col=1)
    # Unphysical + metastable portion (dotted)
    fig.add_trace(go.Scatter(x=v_plot[mask_mid], y=P_data[mask_mid], mode='lines',
                             line=dict(color=C_ISOTHERM, width=2, dash='dot'), showlegend=False),
                  row=2, col=1)
    # Stable portion (gas side)
    fig.add_trace(go.Scatter(x=v_plot[mask_right], y=P_data[mask_right], mode='lines',
                             line=dict(color=C_ISOTHERM, width=2), showlegend=False),
                  row=2, col=1)
    # Maxwell construction: horizontal line at equilibrium pressure
    fig.add_trace(go.Scatter(x=[v_liq, v_gas], y=[P_maxwell, P_maxwell], mode='lines',
                             line=dict(color=C_ISOTHERM, width=2), showlegend=False),
                  row=2, col=1)
else:
    fig.add_trace(go.Scatter(x=v_plot, y=P_data, mode='lines',
                             line=dict(color=C_ISOTHERM, width=2), showlegend=False),
                  row=2, col=1)

# Isobar
fig.add_trace(go.Scatter(x=[v_min, v_max], y=[pressure, pressure],
                         mode='lines', line=dict(color='#64748b', width=1.5, dash='dash'),
                         showlegend=False), row=2, col=1)

# Isotherm dots
for i in range(len(roots)):
    fig.add_trace(go.Scatter(x=[roots[i]], y=[pressure], mode='markers',
                             marker=dict(size=12, color=LABEL_COLOR[labels[i]],
                                       line=dict(color='white', width=2)),
                             name=labels[i], showlegend=True), row=2, col=1)

# Update axes
fig.update_xaxes(title_text="v (volume per particle)", range=[v_min, v_max], fixedrange=True, row=1, col=1)

# Dynamic y-range for βG plot
if mask.any():
    ymin_G, ymax_G = G_data[mask].min(), G_data[mask].max()
    span = np.log10(ymax_G) - np.log10(max(ymin_G, 1e-30))
    y_lower = ymin_G * 10 ** (-0.15 * span)
    y_upper = ymax_G * 10 ** (0.25 * span)
    fig.update_yaxes(title_text="β G(v)", type="log", range=[np.log10(y_lower), np.log10(y_upper)], fixedrange=True, row=1, col=1)
else:
    fig.update_yaxes(title_text="β G(v)", type="log", fixedrange=True, row=1, col=1)

fig.update_xaxes(title_text="v", range=[v_min, v_max], fixedrange=True, row=2, col=1)

# Dynamic y-range for P plot
P_min_data = P_data.min()
P_max_data = P_data.max()

# Use reasonable bounds: ignore extremely negative or positive values
P_y_lower = max(P_min_data, -10)  # Don't go too negative
P_y_upper = min(P_max_data, 10)  # Cap very high values

# Ensure current pressure is visible
P_y_lower = min(P_y_lower, pressure * 0.8)
P_y_upper = max(P_y_upper, pressure * 1.2)

# Add padding
P_span = P_y_upper - P_y_lower
P_y_lower = P_y_lower - 0.05 * P_span
P_y_upper = P_y_upper + 0.05 * P_span

fig.update_yaxes(title_text="P", range=[P_y_lower, P_y_upper], fixedrange=True, row=2, col=1)

# Update layout
fig.update_layout(
    height=900,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    hovermode='closest',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True, config={
    'modeBarButtonsToRemove': ['autoScale2d', 'resetScale2d', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'select2d', 'lasso2d'],
    'scrollZoom': False,
    'doubleClick': False
})

# Explanations
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Top Panel: Free Energy Landscape")
    st.markdown(r"""
    Shows $\beta G(v)$ on a logarithmic scale. At fixed $P$ and $\beta$ (inverse temperature),
    the system seeks to minimize $G$.

    - **Local minima** correspond to stable or metastable phases (liquid and gas)
    - **Local maxima** indicate unstable states
    - **Global minimum** is the truly stable phase
    """)

with col2:
    st.markdown("### 📈 Bottom Panel: Van der Waals Isotherm")
    st.markdown(r"""
    The classic $P$-$v$ diagram showing the Van der Waals equation of state.

    - **Dashed line**: Isobar at pressure $P$
    - **Intersection points**: Possible volumes at this pressure
    - **Colors**: Indicate stability (stable/metastable/unstable)
    """)

st.divider()

st.markdown("### 🔬 Key Physics Insights")

with st.expander("📌 Phase Transition", expanded=True):
    st.markdown(r"""
    At low temperatures (high $\beta$), you'll observe **three intersection points**:

    - **Leftmost** (small $v$): Liquid phase - stable or metastable
    - **Middle** (intermediate $v$): Mechanically unstable ($\partial P/\partial v > 0$)
    - **Rightmost** (large $v$): Gas phase - stable or metastable

    The stable phase has the lowest Gibbs free energy $G$.
    
    Try adjusting the pressure slider to find where the green dots (stable)
    and orange dots (metastable) switch!
    """)

with st.expander("⚖️ Maxwell Construction"):
    st.markdown(r"""
    The actual **phase transition** occurs where the two phases (liquid and gas)
    have **equal Gibbs free energy**. This implements the famous "equal areas"
    rule on the $P$-$v$ diagram.

    Try adjusting the pressure slider to find where the green dots (stable)
    and orange dots (metastable) switch!
    """)

with st.expander("🌡️ Critical Point"):
    st.markdown(rf"""
    As you **decrease $\beta$** from a large value (increase temperature), the three roots get closer together.

    At the **critical point** ($\beta_c = {beta_c:.3f}$, $P_c = {P_c:.3f}$), the distinction
    between liquid and gas vanishes - they become a single **supercritical fluid**.

    Click the "Set to Critical Point" button above to jump to these values! Slide around these values to see fluid smoothly turning into gas.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <small>
    Interactive visualization for PHY 6536 • Van der Waals Equation & Mean-Field Theory<br>
    Built with Streamlit & Plotly
    </small>
</div>
""", unsafe_allow_html=True)
