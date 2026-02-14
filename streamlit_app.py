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

# Isotherm
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
fig.update_xaxes(title_text="v (volume per particle)", range=[v_min, v_max], row=1, col=1)
fig.update_yaxes(title_text="β G(v)", type="log", row=1, col=1)
fig.update_xaxes(title_text="v", range=[v_min, v_max], row=2, col=1)
fig.update_yaxes(title_text="P", range=[P_lo, P_hi], row=2, col=1)

# Update layout
fig.update_layout(
    height=900,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    hovermode='closest',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

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
