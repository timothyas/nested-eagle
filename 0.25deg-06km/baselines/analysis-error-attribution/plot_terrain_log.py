"""
Relate bias and var_diff (HRRR - GFS) to the *log* of the terrain difference
(elevation and slope), instead of the raw / absolute value.

Why: the terrain difference is heavily concentrated near zero with a long tail
(79% of stations |orog_diff| < 50 m). A raw-space Pearson is therefore leverage-
inflated (see Finding 1). Log-scaling the predictor spreads the dense core and
compresses the tail, so we can ask two things the raw plots cannot:

  1. Is the dependence *logarithmic* in terrain magnitude? -> does a linear fit in
     log-space (Pearson_log) describe the cloud better than in raw space?
  2. Is the raw-space correlation just tail-leverage? -> if so, Pearson collapses
     toward Spearman once we log the x-axis (Spearman is transform-invariant, so
     it is the same in both spaces and is the leverage-free reference).

Transforms:
  bias depends on *signed* terrain   -> signed log  slog(x)=sign(x)*log10(1+|x|)
  var_diff depends on *magnitude*     -> log10(|x|) with a small floor

Output: terrain_log_gfs_vs_hrrr.png  (+ a Pearson raw-vs-log table to stdout)
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from plot_terrain_relation import binned_stat, FIELDS
from datasets import parse_dataset, DEFAULT, TOPO

OROG_FLOOR = 1.0    # m;  sub-metre elevation diffs are noise
SLOPE_FLOOR = 0.1   # m/km


def slog(x):
    """Signed log: keeps direction, compresses magnitude. 0 -> 0."""
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def lmag(x, floor):
    return np.log10(np.maximum(np.abs(x), floor))


def load_frame(ds=DEFAULT):
    m = xr.open_dataset(ds["metrics"])
    t = xr.open_dataset(TOPO)
    common = pd.DataFrame({
        "orog_diff": t["orog_diff"].values,             # m
        "slope_diff": t["slope_diff"].values * 1000.0,  # m/m -> m/km
    })
    out = {}
    for f in FIELDS:
        s = m.sel(field=f)
        df = pd.concat([
            pd.DataFrame({"bias": s["bias"].values,
                          "var_diff": s["var_diff"].values}),
            common], axis=1).dropna()
        out[f] = df
    return out


def loghist_panel(ax, x, y, xlabel, ylabel, ref_rho=None):
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    xr_ = np.percentile(x, [0.5, 99.5])
    yr_ = np.percentile(y, [0.5, 99.5])
    h = ax.hist2d(x, y, bins=45, range=[xr_, yr_], norm=LogNorm(), cmap="mako")
    lin = stats.linregress(x, y)
    rho = stats.spearmanr(x, y).statistic
    xs = np.array(xr_)
    ax.plot(xs, lin.intercept + lin.slope * xs, "r-", lw=2,
            label=f"OLS  r_log={lin.rvalue:+.2f}")
    bs = binned_stat(x, y)
    if bs.size:
        ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                label="binned median")
    # Spearman is transform-invariant: same in raw and log space -> leverage-free ref
    ax.plot([], [], " ", label=f"$\\rho$={rho:+.2f} (transform-free)")
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_xlim(xr_)
    ax.set_ylim(yr_)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=7, framealpha=0.8)


def main(ds=DEFAULT):
    frames = load_frame(ds)
    nrow = len(FIELDS)
    fig, axes = plt.subplots(nrow, 4, figsize=(18, 4.4 * nrow),
                             constrained_layout=True)
    for i, (f, (lab, unit, _)) in enumerate(FIELDS.items()):
        df = frames[f]
        loghist_panel(axes[i, 0], slog(df["orog_diff"].values), df["bias"].values,
                      "signed log10(1+|elev diff|) [m]", f"{lab} bias [{unit}]")
        loghist_panel(axes[i, 1], slog(df["slope_diff"].values), df["bias"].values,
                      "signed log10(1+|slope diff|) [m/km]", f"{lab} bias [{unit}]")
        loghist_panel(axes[i, 2], lmag(df["orog_diff"].values, OROG_FLOOR),
                      df["var_diff"].values,
                      "log10|elev diff| [m]", f"{lab} var(HRRR-GFS) [{unit}$^2$]")
        loghist_panel(axes[i, 3], lmag(df["slope_diff"].values, SLOPE_FLOOR),
                      df["var_diff"].values,
                      "log10|slope diff| [m/km]", f"{lab} var(HRRR-GFS) [{unit}$^2$]")
    fig.suptitle(f"{ds['diff']} bias and var_diff vs LOG terrain difference "
                 f"(n={len(frames['2m_temperature'])} stations, {ds['diff']})",
                 fontsize=14)
    out = f"terrain_log_{ds['tag']}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}\n")

    # raw-vs-log Pearson, with the transform-invariant Spearman as anchor
    print(f"{'field':16s} {'pair':22s} {'Pearson_raw':>12s} {'Pearson_log':>12s} "
          f"{'Spearman':>9s}")
    for f in FIELDS:
        df = frames[f]
        specs = [
            ("bias", "orog_diff", df["orog_diff"].values, slog, "bias~elev"),
            ("bias", "slope_diff", df["slope_diff"].values, slog, "bias~slope"),
            ("var_diff", "orog_diff", df["orog_diff"].values,
             lambda v: lmag(v, OROG_FLOOR), "var~|elev|"),
            ("var_diff", "slope_diff", df["slope_diff"].values,
             lambda v: lmag(v, SLOPE_FLOOR), "var~|slope|"),
        ]
        for ycol, _, xv, tf, tag in specs:
            y = df[ycol].values
            xraw = np.abs(xv) if ycol == "var_diff" else xv
            r_raw = stats.pearsonr(xraw, y).statistic
            r_log = stats.pearsonr(tf(xv), y).statistic
            # Spearman anchor must rank the SAME quantity the log transform is
            # monotonic in: |x| for the magnitude (var) rows, signed x for bias.
            rho = stats.spearmanr(xraw, y).statistic
            print(f"{f:16s} {tag:22s} {r_raw:>+12.2f} {r_log:>+12.2f} {rho:>+9.2f}")


if __name__ == "__main__":
    main(parse_dataset(__doc__))
