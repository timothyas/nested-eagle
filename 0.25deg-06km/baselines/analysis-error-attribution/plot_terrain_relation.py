"""
Relate the GFS<->HRRR analysis differences (bias, RMSE at obs stations) to how
differently the two models resolve the terrain.

We pair the per-station metrics (gfs_vs_hrrr.metrics.nc) with the per-station
terrain differences (gfs_vs_hrrr.topo.nc), all on the HRRR-minus-GFS convention:

  orog_diff  = HRRR elevation - GFS elevation   (m;  <0 => HRRR sits lower)
  slope_diff = HRRR |grad h|   - GFS |grad h|    (m/km; >0 => HRRR steeper)

For each field (2 m T, 10 m wind speed) we show:
  * bias vs the *signed* terrain difference -- a directional statement (does HRRR
    being lower/steeper than GFS push the analysis warm/cold, fast/slow?). For
    2 m T vs elevation this should roughly track the average-atmosphere lapse
    rate, so we overlay a -6.5 K/km reference line and report the fitted rate.
  * RMSE vs the *magnitude* |terrain difference| -- disagreement should grow
    where the two models' terrain disagrees most, regardless of sign.

Each panel: 2-D histogram of the cloud, a quantile-binned median (robust trend),
and an OLS fit with Pearson r and (sign-free) Spearman rho annotated.

Output: terrain_relation_gfs_vs_hrrr.png
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from datasets import parse_dataset, DEFAULT, TOPO

# field -> (short label, unit, average-atmosphere reference lapse rate K/km or None)
FIELDS = {
    "2m_temperature": ("2 m T", "K", 6.5),
    "10m_wind_speed": ("10 m wind speed", "m/s", None),
}
# 6.5 K/km: the average tropospheric lapse rate (ICAO/US average atmosphere). A
# loose yardstick for the fit, NOT the dry-adiabatic (9.8) or moist (~5) rate.
LAPSE_REF = 6.5


def load_frame(ds=DEFAULT):
    """One tidy DataFrame: per-station metrics + terrain diffs (m, m/km)."""
    m = xr.open_dataset(ds["metrics"])
    t = xr.open_dataset(TOPO)
    rows = {}
    for f in FIELDS:
        s = m.sel(field=f)
        rows[f] = pd.DataFrame({
            "bias": s["bias"].values,
            "rmse": s["rmse"].values,
        })
    common = pd.DataFrame({
        "orog_diff": t["orog_diff"].values,            # m
        "slope_diff": t["slope_diff"].values * 1000.0,  # m/m -> m/km
    })
    return {f: pd.concat([rows[f], common], axis=1).dropna() for f in FIELDS}


def binned_stat(x, y, nbins=14):
    """Quantile-binned median (+IQR) of y vs x at bin x-medians."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 8:
            continue
        out.append((np.median(x[sel]), np.median(y[sel]),
                    np.quantile(y[sel], 0.25), np.quantile(y[sel], 0.75)))
    return np.array(out).T if out else np.empty((4, 0))


def panel(ax, df, xcol, ycol, xlabel, ylabel, lapse_ref=False):
    x, y = df[xcol].values, df[ycol].values
    sns.histplot(x=x, y=y, bins=45, cmap="mako", ax=ax, cbar=False)
    lin = stats.linregress(x, y)
    rho = stats.spearmanr(x, y).statistic
    xs = np.array([np.quantile(x, 0.005), np.quantile(x, 0.995)])
    ax.plot(xs, lin.intercept + lin.slope * xs, "r-", lw=2,
            label=f"OLS r={lin.rvalue:+.2f}  $\\rho$={rho:+.2f}")
    bs = binned_stat(x, y)
    if bs.size:
        ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                label="binned median")
    if lapse_ref:  # reference dT/dz = -lapse rate; slope here is per metre
        ax.plot(xs, lin.intercept - (LAPSE_REF / 1000.0) * xs, "--",
                color="cyan", lw=1.5,
                label=f"-{LAPSE_REF:.1f} K/km avg-atmosphere ref")
        ax.set_title(f"fitted lapse rate {-lin.slope * 1000:.1f} K/km", fontsize=9)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=7, framealpha=0.75)


def main(ds=DEFAULT):
    frames = load_frame(ds)
    nrow = len(FIELDS)
    fig, axes = plt.subplots(nrow, 4, figsize=(18, 4.2 * nrow),
                             constrained_layout=True)
    for i, (f, (lab, unit, _)) in enumerate(FIELDS.items()):
        df = frames[f]
        df_abs = df.assign(orog_abs=df["orog_diff"].abs(),
                           slope_abs=df["slope_diff"].abs())
        # bias vs signed terrain difference
        panel(axes[i, 0], df, "orog_diff", "bias",
              "HRRR - GFS elevation [m]", f"{lab} bias [{unit}]",
              lapse_ref=(f == "2m_temperature"))
        panel(axes[i, 1], df, "slope_diff", "bias",
              "HRRR - GFS slope [m/km]", f"{lab} bias [{unit}]")
        # RMSE vs magnitude of terrain difference
        panel(axes[i, 2], df_abs, "orog_abs", "rmse",
              "|HRRR - GFS elevation| [m]", f"{lab} RMSE [{unit}]")
        panel(axes[i, 3], df_abs, "slope_abs", "rmse",
              "|HRRR - GFS slope| [m/km]", f"{lab} RMSE [{unit}]")
    fig.suptitle(f"{ds['diff']} {ds['kind']} differences vs terrain representation "
                 f"(n={len(frames['2m_temperature'])} stations, "
                 f"{ds['diff']})", fontsize=14)
    out = f"terrain_relation_{ds['tag']}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")

    # quick numeric summary to stdout
    print("\nPearson r / Spearman rho:")
    for f in FIELDS:
        df = frames[f]
        for ycol, xv, xn in [
            ("bias", df["orog_diff"], "orog_diff"),
            ("bias", df["slope_diff"], "slope_diff"),
            ("rmse", df["orog_diff"].abs(), "|orog_diff|"),
            ("rmse", df["slope_diff"].abs(), "|slope_diff|"),
        ]:
            r = stats.pearsonr(xv, df[ycol]).statistic
            rho = stats.spearmanr(xv, df[ycol]).statistic
            print(f"  {f:16s} {ycol:4s} vs {xn:12s}: r={r:+.2f} rho={rho:+.2f}")


if __name__ == "__main__":
    main(parse_dataset(__doc__))
