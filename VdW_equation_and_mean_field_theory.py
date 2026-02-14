"""
Van der Waals Equation & Mean-Field Theory — Interactive Visualization
======================================================================

Two sliders:
  β  — inverse temperature (changes isotherm shape & free-energy landscape)
  P  — pressure (moves isobar + dots slide along the isotherm)

Stability classification (color-coded on both panels):
  ● green  — stable    (global min of G  ↔  lowest local min of G)
  ● orange — metastable (higher local min of G)
  ● red    — unstable   (local max of G, saddle point)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Model parameters ────────────────────────────────────────────────────
Omega = 0.02        # hard-core volume  (Ω)
U0 = 1.5            # interaction strength
u = Omega * U0      # effective interaction
a = u / 2.0         # VdW attraction
b = Omega / 2.0     # VdW excluded volume

# ── Plotting ranges ─────────────────────────────────────────────────────
v_min = b + 1e-6
v_max = 0.4
P_lo, P_hi = 0.2, 10.0

v_fine = np.linspace(v_min, v_max, 4000)   # fine grid for root finding
v_plot = np.linspace(v_min, v_max, 600)    # plotting grid

# ── Colors ──────────────────────────────────────────────────────────────
C_STABLE    = "#16a34a"   # green
C_METASTABLE = "#f59e0b"  # amber
C_UNSTABLE  = "#dc2626"   # red
C_ISOTHERM  = "#2563eb"   # blue

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
    """
    Classify each root as 'stable', 'metastable', or 'unstable'.

    At a root v*:
      d²(−βG)/dv² > 0  →  local min of G  →  (meta)stable equilibrium
      d²(−βG)/dv² < 0  →  local max of G  →  unstable (saddle point)

    Among the (meta)stable ones, the one with the lowest G is globally
    stable; the one with higher G is metastable.
    """
    if len(roots) == 0:
        return []

    d2 = d2_neg_beta_G(roots, beta)
    G_vals = neg_beta_G(roots, beta, pressure)

    labels = []
    for i in range(len(roots)):
        if d2[i] > 0:          # local min of G  →  stable/metastable
            labels.append("_stable_candidate")
        else:
            labels.append("unstable")

    # Among the stable candidates, pick the global winner (lowest G)
    candidates = [(i, G_vals[i]) for i, l in enumerate(labels)
                  if l == "_stable_candidate"]

    if len(candidates) == 1:
        labels[candidates[0][0]] = "stable"
    elif len(candidates) >= 2:
        best_idx = min(candidates, key=lambda x: x[1])[0]  # lowest G
        for i, _ in candidates:
            labels[i] = "stable" if i == best_idx else "metastable"

    return labels


LABEL_COLOR = {
    "stable":     C_STABLE,
    "metastable": C_METASTABLE,
    "unstable":   C_UNSTABLE,
}

# ── Initial state ───────────────────────────────────────────────────────
beta_init = 2.5
p_init = 2.3

# ── Set up figure ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
fig.subplots_adjust(bottom=0.25, wspace=0.32, left=0.07, right=0.97, top=0.92)
fig.suptitle("Van der Waals Equation  &  Mean-Field Theory",
             fontsize=13, fontweight="bold")

# ── Left panel: −βG(v) ─────────────────────────────────────────────────
y0 = neg_beta_G(v_plot, beta_init, p_init)
mask0 = y0 > 0
line_G, = ax1.semilogy(v_plot[mask0], y0[mask0], color=C_ISOTHERM, lw=2)
ax1.set_xlabel("v  (volume per particle)", fontsize=11)
ax1.set_ylabel(r"$\beta\, G(v)$", fontsize=12)
ax1.set_xlim(v_min, v_max)
ax1.set_title("Free-energy landscape (log scale)", fontsize=11)
text_P_left = ax1.text(0.02, 0.98, f"P = {p_init:.2f}", transform=ax1.transAxes,
                       fontsize=11, fontweight="bold", va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
text_beta_left = ax1.text(0.02, 0.92, rf"$\beta$ = {beta_init:.2f}", transform=ax1.transAxes,
                          fontsize=11, fontweight="bold", va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

# Legend for left panel
from matplotlib.lines import Line2D
legend_handles_left = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_STABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="stable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_METASTABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="metastable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UNSTABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="unstable (saddle)"),
]
ax1.legend(handles=legend_handles_left, loc="upper right", fontsize=8.5,
           framealpha=0.9)

# Placeholders for up to 3 dots on the free-energy plot
G_dots = []
for _ in range(3):
    d, = ax1.plot([], [], "o", ms=10, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.5)
    d.set_visible(False)
    G_dots.append(d)

# ── Right panel: VdW isotherm + isobar + dots ──────────────────────────
P_iso0 = vdw_pressure(v_plot, beta_init)
# Split isotherm into stable (solid) and unstable/metastable (dashed) segments
line_iso_left, = ax2.plot([], [], color=C_ISOTHERM, lw=2)
line_iso_mid, = ax2.plot([], [], color=C_ISOTHERM, lw=2, ls=':')
line_iso_right, = ax2.plot([], [], color=C_ISOTHERM, lw=2)
line_maxwell, = ax2.plot([], [], color=C_ISOTHERM, lw=2)
line_isobar = ax2.axhline(p_init, color="#64748b", ls="--", lw=1.2, alpha=0.6)

# Placeholders for up to 3 dots + vlines on the isotherm plot
iso_dots = []
iso_vlines = []
for _ in range(3):
    d, = ax2.plot([], [], "o", ms=10, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.5)
    d.set_visible(False)
    iso_dots.append(d)
    vl = ax2.axvline(0, color="gray", ls=":", lw=1, alpha=0)
    iso_vlines.append(vl)

# Legend for right panel
legend_handles_right = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_STABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="stable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_METASTABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="metastable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UNSTABLE,
           markeredgecolor="white", markeredgewidth=1.5, ms=10, label="unstable (saddle)"),
]
ax2.legend(handles=legend_handles_right, loc="upper right", fontsize=8.5,
           framealpha=0.9)

ax2.set_xlabel("v", fontsize=11)
ax2.set_ylabel("P", fontsize=11)
ax2.set_xlim(v_min, v_max)
ax2.set_ylim(P_lo, P_hi)
ax2.set_title("VdW isotherm  &  isobar", fontsize=11)
# P label positioned near the isobar line
text_P_right = ax2.text(v_max * 0.95, p_init, f"P = {p_init:.2f}",
                        fontsize=10, fontweight="bold", va="center", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#64748b", alpha=0.9))
text_beta_right = ax2.text(0.02, 0.98, rf"$\beta$ = {beta_init:.2f}", transform=ax2.transAxes,
                           fontsize=11, fontweight="bold", va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

# ── Sliders ─────────────────────────────────────────────────────────────
ax_beta = fig.add_axes([0.20, 0.10, 0.60, 0.03])
ax_pres = fig.add_axes([0.20, 0.04, 0.60, 0.03])

slider_beta = Slider(ax_beta, r"$\beta$", 1.5, 4.0,
                     valinit=beta_init, valstep=0.01, color="#6366f1")
slider_pres = Slider(ax_pres, r"$P$", P_lo, P_hi,
                     valinit=p_init, valstep=0.05, color="#94a3b8")


def update(_=None):
    beta = slider_beta.val
    pressure = slider_pres.val

    # Update text labels
    text_P_left.set_text(f"P = {pressure:.2f}")
    text_beta_left.set_text(rf"$\beta$ = {beta:.2f}")
    text_beta_right.set_text(rf"$\beta$ = {beta:.2f}")
    text_P_right.set_text(f"P = {pressure:.2f}")
    text_P_right.set_position((v_max * 0.95, pressure))

    # ── Left panel: free-energy curve ───────────────────────────────────
    y = neg_beta_G(v_plot, beta, pressure)
    mask = y > 0
    line_G.set_xdata(v_plot[mask])
    line_G.set_ydata(y[mask])
    if mask.any():
        ymin, ymax = y[mask].min(), y[mask].max()
        span = np.log10(ymax) - np.log10(max(ymin, 1e-30))
        ax1.set_ylim(ymin * 10 ** (-0.15 * span), ymax * 10 ** (0.25 * span))

    # ── Find & classify roots ───────────────────────────────────────────
    roots = find_roots(beta, pressure)
    labels = classify_roots(roots, beta, pressure)
    G_at_roots = neg_beta_G(roots, beta, pressure) if len(roots) > 0 else []

    # ── Right panel: isotherm ───────────────────────────────────────────
    P_iso = vdw_pressure(v_plot, beta)
    line_isobar.set_ydata([pressure, pressure])

    # Split isotherm via Maxwell construction (depends only on beta)
    P_mw, v_liq, v_gas = find_maxwell_pressure(beta)
    if P_mw is not None:
        mask_left = v_plot <= v_liq
        mask_mid = (v_plot >= v_liq) & (v_plot <= v_gas)
        mask_right = v_plot >= v_gas
        line_iso_left.set_data(v_plot[mask_left], P_iso[mask_left])
        line_iso_mid.set_data(v_plot[mask_mid], P_iso[mask_mid])
        line_iso_right.set_data(v_plot[mask_right], P_iso[mask_right])
        line_maxwell.set_data([v_liq, v_gas], [P_mw, P_mw])
    else:
        line_iso_left.set_data(v_plot, P_iso)
        line_iso_mid.set_data([], [])
        line_iso_right.set_data([], [])
        line_maxwell.set_data([], [])

    # ── Update dots (both panels) ───────────────────────────────────────
    for i in range(3):
        if i < len(roots):
            c = LABEL_COLOR[labels[i]]
            v_r = roots[i]
            G_r = G_at_roots[i]

            # Left panel dot
            G_dots[i].set_xdata([v_r])
            G_dots[i].set_ydata([G_r])
            G_dots[i].set_color(c)
            G_dots[i].set_visible(G_r > 0)

            # Right panel dot
            iso_dots[i].set_xdata([v_r])
            iso_dots[i].set_ydata([pressure])
            iso_dots[i].set_color(c)
            iso_dots[i].set_visible(True)
            iso_vlines[i].set_xdata([v_r, v_r])
            iso_vlines[i].set_color(c)
            iso_vlines[i].set_alpha(0.4)
        else:
            G_dots[i].set_visible(False)
            iso_dots[i].set_visible(False)
            iso_vlines[i].set_alpha(0)

    fig.canvas.draw_idle()


slider_beta.on_changed(update)
slider_pres.on_changed(update)

# Draw initial state
update()

plt.show()