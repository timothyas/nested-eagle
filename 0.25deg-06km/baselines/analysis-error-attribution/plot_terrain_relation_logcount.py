"""
Log-count version of the bias-vs-terrain bivariate histograms.

The linear-count histograms (plot_terrain_relation.py) saturate on the dense
near-origin core (~79% of stations have |orog_diff| < 50 m) and render the
sparse, high-leverage tail as an indistinguishable black void -- which hides
that the global Pearson r is carried by that tail (r=-0.66 full sample but only
-0.09 within the core; Spearman rho=-0.35).

Here we use a logarithmic colour normalization so each decade of bin count is
visible, and we annotate the honest summaries: Spearman rho (rank, leverage-
robust) and the Theil-Sen slope (robust regression) alongside the leverage-
sensitive Pearson r / OLS. The binned median makes the (real, physical) trend
legible without depending on the tail.

Output: terrain_relation_logcount_gfs_vs_hrrr.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from plot_terrain_relation import load_frame, binned_stat, FIELDS, LAPSE_REF


def panel(ax, x, y, xlabel, ylabel, lapse_ref=False):
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    # display extent: clip the axis to robust percentiles so a handful of extreme
    # outliers don't blow out the core, while log-count still reveals the tail.
    xr = np.percentile(x, [0.5, 99.5])
    yr = np.percentile(y, [0.5, 99.5])
    h = ax.hist2d(x, y, bins=45, range=[xr, yr], norm=LogNorm(), cmap="mako")
    cb = plt.colorbar(h[3], ax=ax, pad=0.01, fraction=0.046)
    cb.set_label("bin count (log)", fontsize=8)

    # fits on ALL data (this is exactly what we are scrutinizing)
    ols = stats.linregress(x, y)
    ts = stats.theilslopes(y, x)        # robust slope/intercept (leverage-resistant)
    rho = stats.spearmanr(x, y).statistic
    xs = np.array(xr)
    ax.plot(xs, ols.intercept + ols.slope * xs, "-", color="red", lw=1.6,
            label=f"OLS  r={ols.rvalue:+.2f}")
    ax.plot(xs, ts[1] + ts[0] * xs, "--", color="lime", lw=1.8,
            label=f"Theil-Sen  $\\rho$={rho:+.2f}")
    bs = binned_stat(x, y)
    if bs.size:
        ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                label="binned median")
    if lapse_ref:
        ax.plot(xs, ols.intercept - (LAPSE_REF / 1000.0) * xs, ":", color="cyan",
                lw=1.5, label=f"-{LAPSE_REF:.1f} K/km avg-atmosphere ref")
        ax.set_title(f"OLS {-ols.slope*1000:.1f} / robust {-ts[0]*1000:.1f} K/km",
                     fontsize=9)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=7, framealpha=0.75)


def main():
    frames = load_frame()
    nrow = len(FIELDS)
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 4.6 * nrow),
                             constrained_layout=True)
    for i, (f, (lab, unit, _)) in enumerate(FIELDS.items()):
        df = frames[f]
        panel(axes[i, 0], df["orog_diff"].values, df["bias"].values,
              "HRRR - GFS elevation [m]", f"{lab} bias [{unit}]",
              lapse_ref=(f == "2m_temperature"))
        panel(axes[i, 1], df["slope_diff"].values, df["bias"].values,
              "HRRR - GFS slope [m/km]", f"{lab} bias [{unit}]")
    fig.suptitle("Bias vs terrain difference, log-count density "
                 f"(n={len(frames['2m_temperature'])} stations; robust fit + "
                 "Spearman shown alongside leverage-sensitive OLS/Pearson)",
                 fontsize=13)
    out = "terrain_relation_logcount_gfs_vs_hrrr.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
