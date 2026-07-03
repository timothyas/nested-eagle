"""Read the effective lapse rate directly from the nested-IC t2m difference.

The forecast difference is ``d = nested-lam - fakehrrr`` (member-mean, then over
t0). ``fc1=nested-lam`` is initialized with the true HRRR orography; ``fc2=fakehrrr``
is initialized with GFS orography conservatively regridded to 6 km. So the static
height misrepresentation seen by the two runs is

    dz = orog_hrrr - orog_gfs      [m]   (positive where HRRR terrain is higher)

If the early t2m difference is dominated by the surface temperature being
diagnosed at different model elevations, pure lapse-rate physics predicts

    d_t2m  ~  -Gamma * dz            with  Gamma ~ 6.5 K/km  (~ -0.0065 K/m).

This script regresses the fhr=0 t2m bias map (= mean_t0 d) on dz and reads the
slope. A slope near -6.5 K/km, falling toward 0 as the ICs converge, would clinch
the lapse-rate / convergence interpretation of the terrain-locked variance.

Run in the `eagle` conda env:
    conda run -n eagle python lapse_rate_test.py
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, linregress, kendalltau, spearmanr

from eagle.tools.data import trim_xarray_edge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("lapse")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
METRICS_DIR = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/production/gfs-hrrr/"
    "stage1c/inference-testing/spatial-metrics-vs-fakehrrr"
)
MODEL_LABEL = "nested-lamvnested-lam.fc1vfc2"

HRRR_OROG = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines/"
    "hrrr-forecasts-vs-hrrr-analysis/hrrr.forecasts.zarr"
)
GFS_OROG = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines/"
    "gfs-forecasts-vs-hrrr-analysis/gfs.forecasts.zarr"
)
OROG_T0 = "2024-02-01T06"  # same model version across t0; one slice suffices

LCC_INFO = {"n_x": 848, "n_y": 480}
TRIM_EDGE = [25, 24, 25, 26]

OUTDIR = "./figures"
METRIC = "bias"
VAR = "sh2"
FHRS = [0, 6, 12, 24, 48]
LAPSE_REF = -6.5e-3  # K/m, standard tropospheric lapse for the reference line


def load_orog(path):
    da = xr.open_zarr(path)["orog"].sel(t0=OROG_T0, method="nearest").load()
    da = da.to_dataset(name="orog")
    return trim_xarray_edge(da, LCC_INFO, TRIM_EDGE, ("y", "x"))["orog"]


def regress(dz_1d, d_1d, ts_sample=8000, seed=0):
    """OLS (full field) + robust Theil-Sen slope of d on dz. Slopes in K/km.

    Theil-Sen is O(n^2) in the pairwise slopes, so it is computed on a random
    subsample; OLS uses every gridcell.
    """
    good = np.isfinite(dz_1d) & np.isfinite(d_1d)
    x, y = dz_1d[good], d_1d[good]
    ols = linregress(x, y)
    if x.size > ts_sample:
        idx = np.random.default_rng(seed).choice(x.size, ts_sample, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    ts = theilslopes(ys, xs)  # (slope, intercept, lo, hi) at 95%
    # Kendall's tau is the rank-concordance partner to the Theil-Sen slope
    # (the slope is the value that zeroes tau between residuals and dz). It is
    # O(n log n) in scipy, so it runs on the full field, not the subsample.
    tau = kendalltau(x, y).statistic
    # Spearman rho^2 is the rank (leverage-free) analog of the OLS r^2 -- the
    # honest 'variance explained' number, also full-field.
    rho = spearmanr(x, y).statistic
    return {
        "ols_slope_km": ols.slope * 1000.0,
        "ols_intercept": ols.intercept,
        "ols_r": ols.rvalue,
        "ts_slope_km": ts[0] * 1000.0,
        "ts_lo_km": ts[2] * 1000.0,
        "ts_hi_km": ts[3] * 1000.0,
        "kendall_tau": tau,
        "spearman_rho2": rho**2,
        "n": int(x.size),
    }


def _scatter_panel(ax, dz_flat, d_flat, fhr):
    """Hexbin of d vs dz at one lead time, with binned median, reference lapse,
    and Theil-Sen lines; title carries the fitted slopes."""
    good = np.isfinite(dz_flat) & np.isfinite(d_flat)
    x, y = dz_flat[good], d_flat[good]
    ax.hexbin(x, y, gridsize=70, bins="log", cmap="Greys", mincnt=1)

    # binned median spine
    edges = np.quantile(x, np.linspace(0, 1, 26))
    edges[-1] += 1e-6
    which = np.digitize(x, edges) - 1
    cx, cy = [], []
    for b in range(25):
        sel = which == b
        if sel.sum() > 50:
            cx.append(np.median(x[sel]))
            cy.append(np.median(y[sel]))
    ax.plot(cx, cy, "-o", color="C3", ms=3, lw=1.5, label="binned median")

    r = regress(dz_flat, d_flat)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, LAPSE_REF * xs, "C0--", lw=1.6,
            label=f"reference {LAPSE_REF*1000:+.1f} K/km")
    ax.plot(xs, (r["ols_slope_km"] / 1000.0) * xs + r["ols_intercept"],
            "C1-", lw=1.6,
            label=f"OLS {r['ols_slope_km']:+.2f} K/km (r = {r['ols_r']:+.2f}, ρ² = {r['spearman_rho2']:.2f})")
    ax.plot(xs, (r["ts_slope_km"] / 1000.0) * xs + np.median(y),
            "C2-", lw=1.6,
            label=f"Theil-Sen {r['ts_slope_km']:+.2f} K/km (τ = {r['kendall_tau']:+.2f})")

    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("dz = orog$_{HRRR}$ - orog$_{GFS}$  [m]")
    ax.legend(title=f"fhr = {fhr:02d}h", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    plt.style.use("~/nice.mplstyle")
    orog_hrrr = load_orog(HRRR_OROG)
    orog_gfs = load_orog(GFS_OROG)
    dz = (orog_hrrr - orog_gfs).rename("dz")  # m, HRRR(true) - GFS(fake)
    dz = dz.transpose("y", "x")
    logger.info(
        f"dz = orog_hrrr - orog_gfs:  mean {float(dz.mean()):+.1f} m, "
        f"std {float(dz.std()):.1f} m, "
        f"range [{float(dz.min()):+.0f}, {float(dz.max()):+.0f}] m"
    )

    metric = xr.open_dataset(
        f"{METRICS_DIR}/spatial.diff.{METRIC}.{MODEL_LABEL}.nc"
    )[VAR]

    dz_flat = dz.values.ravel()
    rows = []
    for fhr in FHRS:
        d = metric.sel(fhr=fhr)
        d = d.transpose("y", "x")
        d = d.values.ravel()
        rows.append({"fhr": fhr, **regress(dz_flat, d)})
    tab = pd.DataFrame(rows)
    logger.info(f"\n=== {VAR} {METRIC} = mean_t0(d) regressed on dz ===")
    logger.info(
        tab[["fhr", "ols_slope_km", "ts_slope_km", "ts_lo_km", "ts_hi_km",
             "ols_r", "kendall_tau", "spearman_rho2", "n"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"
        )
    )
    logger.info(f"\nreference lapse rate: {LAPSE_REF*1000:+.2f} K/km")
    tab.to_csv(f"{OUTDIR}/lapse_rate_slope.csv", index=False)

    # --- scatter panels for several lead times ------------------------------
    plot_fhrs = [0, 6, 24]
    fig, axes = plt.subplots(
        1, len(plot_fhrs), figsize=(5.0 * len(plot_fhrs), 5.2),
        sharex=True, sharey=True,
    )
    for ax, fhr in zip(np.atleast_1d(axes), plot_fhrs):
        _scatter_panel(ax, dz_flat, metric.sel(fhr=fhr).values.ravel(), fhr)
    axes[0].set_ylabel(
        f"{VAR} {METRIC} = mean$_{{t0}}$(nested-lam - fakehrrr)  [K]"
    )
    fig.suptitle(
        f"{VAR} forecast difference vs orography misrepresentation "
        f"(dz = orog$_{{HRRR}}$ - orog$_{{GFS}}$)"
    )
    fig.tight_layout()
    out = f"{OUTDIR}/lapse_rate_{VAR}-{METRIC}_vs_dz.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info(f"\nwrote {out}")


if __name__ == "__main__":
    main()
