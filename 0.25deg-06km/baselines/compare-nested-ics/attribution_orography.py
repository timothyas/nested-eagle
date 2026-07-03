"""Orography attribution of the nested-IC forecast-difference error metrics.

We have spatial maps of the bias and variance of the per-IC forecast difference
``d = nested-lam - fakehrrr`` (member-mean, then over the collection of initial
conditions ``t0``), as a function of lead time ``fhr``:

    bias(x)     = mean_t0[d]                       spatial.diff.bias.*.nc
    variance(x) = var_t0[d]   (MSE = bias^2 + var) spatial.diff.variance.*.nc

This script asks: **how much of the spatial structure of those maps is explained
by the static orography** (and, secondarily, terrain slope |grad orog|)? The
"variance explained" is the spatial R^2 of the metric map regressed on the
predictor, with gridcells as the sample.

Methodology (carried from baselines/analysis-error-attribution/gfs_vs_hrrr_writeup.md):

  * Report each correlation THREE ways:
      - Pearson on the raw predictor,
      - Pearson on a log-scaled predictor  sign(x)*log10(1+|x|),
      - Spearman rho (rank-based; identical under any monotonic transform).
    The pattern  raw >> log ~ rank  means the correlation is *leverage-inflated*
    by the sparse high-terrain tail; raw ~ log ~ rank means a *robust bulk*
    relationship. The leverage-free headline is rho^2, not raw R^2.

  * Everything is computed per lead time -> "variance explained vs lead time".

Significance notes (the part worth keeping in mind):

  * With ~4e5 gridcells, EVERYTHING is formally significant. Effect size
    (R^2 / rho^2), not a p-value, is the limiter.
  * Worse than the station study: gridcells are heavily spatially autocorrelated,
    so the effective sample size is << N and naive p-values are wildly
    overconfident. We therefore quantify uncertainty with a SPATIAL BLOCK
    BOOTSTRAP (resample contiguous tiles, not individual cells), which respects
    the autocorrelation. Reported for the t2m headline.

The LCC 6 km cells are ~equal area, so correlations are unweighted.

Structure note: predictors live in a dict and the per-(var, fhr) loop is generic,
so adding time-based predictors later (valid_hour / season from the per-IC zarr)
is a drop-in -- swap/extend `build_predictors`.

Run in the `eagle` conda env:
    conda run -n eagle python attribution_orography.py
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from eagle.tools.data import trim_xarray_edge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("attribution")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
METRICS_DIR = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/production/gfs-hrrr/"
    "stage1c/inference-testing/spatial-metrics-vs-fakehrrr"
)
MODEL_LABEL = "nested-lamvnested-lam.fc1vfc2"
OROG_ZARR = "/pscratch/sd/t/timothys/nested-eagle/case-studies/la-fires/hrrr.forecasts.zarr"

# Trim that maps the HRRR (529 x 899) grid onto the (480 x 848) nested-LAM grid.
LCC_INFO = {"n_x": 848, "n_y": 480}
TRIM_EDGE = [25, 24, 25, 26]

OUTDIR = "."
HEADLINE_VAR = "t2m"
HEADLINE_FHRS = [0, 24, 48]
N_BOOT = 1000
BLOCK = 48  # gridcells per side of a bootstrap tile (~288 km at 6 km spacing)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_metric(name):
    """Open one diff metric file (name in {bias, variance, rmse})."""
    path = f"{METRICS_DIR}/spatial.diff.{name}.{MODEL_LABEL}.nc"
    return xr.open_dataset(path)


def build_predictors():
    """Static terrain predictors on the diff grid: orography and slope |grad h|.

    Slope is the gradient magnitude in m per km (6 km spacing); the absolute
    scaling is irrelevant to correlations but m/km is interpretable.
    """
    orog = (
        xr.open_zarr(OROG_ZARR)["orog"]
        .isel(t0=0)
        .load()
        .to_dataset(name="orog")
    )
    orog = trim_xarray_edge(orog, LCC_INFO, TRIM_EDGE, ("y", "x"))["orog"]

    dz_dy, dz_dx = np.gradient(orog.values)  # m per gridcell along y, x
    slope = np.hypot(dz_dx, dz_dy) / 6.0  # -> m/km
    slope = xr.DataArray(slope, dims=orog.dims, coords=orog.coords)

    return {"orog": orog, "slope": slope}


# ----------------------------------------------------------------------------
# Correlation diagnostics
# ----------------------------------------------------------------------------
def _log_scale(x):
    """Leverage-compressing transform that preserves sign: sign(x)*log10(1+|x|)."""
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def correlations(metric_1d, predictor_1d):
    """Raw/log Pearson and Spearman between a metric map and a predictor map.

    Returns a dict with the three correlations plus R^2 (raw Pearson^2) and
    rho^2 (Spearman^2 -- the leverage-free 'variance explained').
    """
    x = np.asarray(predictor_1d, dtype=float)
    y = np.asarray(metric_1d, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    pear_raw = pearsonr(x, y).statistic
    pear_log = pearsonr(_log_scale(x), y).statistic
    spear = spearmanr(x, y).statistic
    return {
        "pearson_raw": pear_raw,
        "pearson_log": pear_log,
        "spearman": spear,
        "R2": pear_raw**2,
        "rho2": spear**2,
        "n": x.size,
    }


def build_table(metrics, predictors):
    """Per (metric, variable, fhr, predictor) correlation table -> tidy DataFrame."""
    rows = []
    for metric_name, ds in metrics.items():
        for var in ds.data_vars:
            for fhr in ds.fhr.values:
                field = ds[var].sel(fhr=fhr).values.ravel()
                for pred_name, pred in predictors.items():
                    stats = correlations(field, pred.values.ravel())
                    rows.append(
                        {
                            "metric": metric_name,
                            "variable": var,
                            "fhr": int(fhr),
                            "predictor": pred_name,
                            **stats,
                        }
                    )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Spatial block bootstrap (honest uncertainty under spatial autocorrelation)
# ----------------------------------------------------------------------------
def block_bootstrap_ci(metric_2d, predictor_2d, block=BLOCK, n_boot=N_BOOT, seed=0):
    """95% CI on Pearson and Spearman via a non-overlapping-tile block bootstrap.

    Tiles of ``block`` x ``block`` gridcells are resampled with replacement, so
    spatial autocorrelation up to the tile scale is preserved (individual-cell
    resampling would pretend the cells are independent and give absurdly tight
    intervals). Spearman is approximated by Pearson on globally-ranked fields so
    the inner loop stays cheap.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(predictor_2d, dtype=float)
    y = np.asarray(metric_2d, dtype=float)
    ny, nx = x.shape

    # Globally-ranked copies for the Spearman approximation.
    finite = np.isfinite(x) & np.isfinite(y)
    xr_ = pd.Series(x[finite]).rank().to_numpy()
    yr_ = pd.Series(y[finite]).rank().to_numpy()
    xr_full = np.full(x.shape, np.nan); xr_full[finite] = xr_
    yr_full = np.full(y.shape, np.nan); yr_full[finite] = yr_

    # Group flattened cell indices by tile.
    iy, ix = np.mgrid[0:ny, 0:nx]
    tile_id = (iy // block) * (nx // block + 1) + (ix // block)
    tiles = [np.flatnonzero(tile_id.ravel() == t) for t in np.unique(tile_id)]

    xf, yf = x.ravel(), y.ravel()
    xrf, yrf = xr_full.ravel(), yr_full.ravel()

    def _corr(idx):
        a, b = xf[idx], yf[idx]
        m = np.isfinite(a) & np.isfinite(b)
        ar, br = xrf[idx][m], yrf[idx][m]
        return np.corrcoef(a[m], b[m])[0, 1], np.corrcoef(ar, br)[0, 1]

    pear, spear = [], []
    n_tiles = len(tiles)
    for _ in range(n_boot):
        pick = rng.integers(0, n_tiles, size=n_tiles)
        idx = np.concatenate([tiles[p] for p in pick])
        p, s = _corr(idx)
        pear.append(p)
        spear.append(s)

    return {
        "pearson_ci": np.percentile(pear, [2.5, 97.5]),
        "spearman_ci": np.percentile(spear, [2.5, 97.5]),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
_SCORES = {
    # column-builder, y-axis label, filename suffix, figure subtitle
    "rho2": (
        lambda t: t["spearman"] ** 2,
        r"variance explained  $\rho^2$  (Spearman$^2$)",
        "",
        "leverage-free",
    ),
    "pearson_log2": (
        lambda t: t["pearson_log"] ** 2,
        r"variance explained  $R^2$  (log-scaled $orog$)",
        "_log",
        "log-scaled predictor",
    ),
}


def plot_variance_explained_vs_leadtime(table, predictor="orog", score="rho2"):
    """Variance-explained vs lead time (bias and variance), for the chosen score.

    score="rho2"         -> Spearman^2 (rank-based, transform-invariant)
    score="pearson_log2" -> R^2 of the metric regressed on log-scaled orography
    """
    builder, ylabel, suffix, subtitle = _SCORES[score]
    sub = table[table.predictor == predictor].copy()
    sub["_y"] = builder(sub)
    variables = sorted(sub.variable.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["bias", "variance"]):
        m = sub[sub.metric == metric]
        for var in variables:
            v = m[m.variable == var].sort_values("fhr")
            ax.plot(v.fhr, v["_y"], marker="o", ms=4, label=var)
        ax.set_title(f"{metric}  ~  {predictor}")
        ax.set_xlabel("lead time [h]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"Orography-explained share of nested-IC difference error ({subtitle})"
    )
    fig.tight_layout()
    out = f"{OUTDIR}/variance_explained_vs_leadtime{suffix}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out}")


def plot_variance_budget(metrics, table, var=HEADLINE_VAR, predictor="orog"):
    """Absolute difference-variance budget vs lead time: terrain-explained vs residual.

    Tests the hypothesis that the terrain-locked part of the t2m difference is a
    fast-decaying IC-misrepresentation transient. The terrain-explained share uses
    R^2 from the log-scaled regression times the domain-mean variance -- a heuristic
    split (R^2 is a spatial-pattern fraction), but the conclusion is also visible in
    the bare R^2(fhr) decay, so it does not hinge on the product.
    """
    var_ds = metrics["variance"][var]
    bias_ds = metrics["bias"][var]
    t = table[(table.variable == var) & (table.predictor == predictor)].set_index(
        ["metric", "fhr"]
    )
    fhrs = var_ds.fhr.values
    tot = np.array([float(var_ds.sel(fhr=f).mean()) for f in fhrs])
    bias_rms = np.array([float(np.sqrt((bias_ds.sel(fhr=f) ** 2).mean())) for f in fhrs])
    r2 = np.array([t.loc[("variance", f), "pearson_log"] ** 2 for f in fhrs])
    terr = r2 * tot

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.fill_between(fhrs, 0, terr, color="C3", alpha=0.55, label="terrain-explained (R$^2_{\\log}$·tot)")
    ax.fill_between(fhrs, terr, tot, color="0.7", alpha=0.7, label="residual variance")
    ax.plot(fhrs, tot, "k-", lw=1.5, label="total diff variance")
    ax.plot(fhrs, bias_rms ** 2, "C0--", lw=1.5, label="bias$^2$ (RMS bias squared)")
    ax.set_xlabel("lead time [h]")
    ax.set_ylabel(f"{var} difference variance  [K$^2$]")
    ax.set_title(f"{var} difference-variance budget vs lead time\n"
                 "terrain-locked part is a fast-decaying transient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = f"{OUTDIR}/{var}_variance_budget_vs_leadtime.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out}")


def plot_headline_scatter(metrics, predictors, var=HEADLINE_VAR, fhr=24, predictor="orog"):
    """Metric vs LOG-scaled predictor + binned-median spine, for bias and variance.

    The x-axis is the leverage-compressing transform ``sign(x)*log10(1+|x|)`` --
    the same one used for the 'Pearson log' diagnostic -- since for fields like
    t2m the metric is linear in log(orography), not in orography itself (raw <
    log ~ rank). Tick labels are placed back in metres for readability.
    """
    pred = predictors[predictor].values.ravel()
    xfull = _log_scale(pred)

    # Tick positions in transformed space, labelled with the original metres.
    tick_m = np.array([0, 100, 500, 1500, 4000])
    tick_pos = _log_scale(tick_m.astype(float))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric in zip(axes, ["bias", "variance"]):
        y = metrics[metric][var].sel(fhr=fhr).values.ravel()
        good = np.isfinite(xfull) & np.isfinite(y)
        x, yy = xfull[good], y[good]
        ax.hexbin(x, yy, gridsize=60, bins="log", cmap="Greys", mincnt=1)

        # binned-median spine over quantiles of the (transformed) predictor
        edges = np.quantile(x, np.linspace(0, 1, 21))
        edges[-1] += 1e-6
        which = np.digitize(x, edges) - 1
        centers, med = [], []
        for b in range(20):
            sel = which == b
            if sel.sum() > 50:
                centers.append(np.median(x[sel]))
                med.append(np.median(yy[sel]))
        ax.plot(centers, med, "-o", color="C3", ms=3, lw=1.5, label="binned median")

        st = correlations(yy, pred[good])
        ax.set_title(
            f"{var} {metric} (fhr={fhr})\n"
            f"Pearson raw/log = {st['pearson_raw']:+.2f}/{st['pearson_log']:+.2f}, "
            f"Spearman = {st['spearman']:+.2f}"
        )
        ax.set_xlabel(f"{predictor} [m] (log-scaled axis)")
        ax.set_ylabel(f"{var} {metric}")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_m)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = f"{OUTDIR}/headline_{var}_vs_{predictor}_log.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    metrics = {"bias": load_metric("bias"), "variance": load_metric("variance")}
    predictors = build_predictors()
    logger.info(f"variables: {list(metrics['bias'].data_vars)}")
    logger.info(f"predictors: {list(predictors)}")

    table = build_table(metrics, predictors)
    csv = f"{OUTDIR}/attribution_orography.csv"
    table.to_csv(csv, index=False)
    logger.info(f"wrote {csv}  ({len(table)} rows)")

    # Console summary: orography, the headline lead times.
    logger.info("\n=== orography correlations (raw / log Pearson, Spearman, rho^2) ===")
    show = table[(table.predictor == "orog") & (table.fhr.isin(HEADLINE_FHRS))]
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        logger.info(
            show.set_index(["metric", "variable", "fhr"])[
                ["pearson_raw", "pearson_log", "spearman", "rho2"]
            ].to_string()
        )

    # Elevation vs slope, headline variable: which predictor wins?
    logger.info(f"\n=== {HEADLINE_VAR}: orography vs slope (Spearman rho) ===")
    comp = table[(table.variable == HEADLINE_VAR) & (table.fhr.isin(HEADLINE_FHRS))]
    logger.info(
        comp.pivot_table(
            index=["metric", "fhr"], columns="predictor", values="spearman"
        ).to_string(float_format=lambda v: f"{v:+.3f}")
    )

    # Figures
    plot_variance_explained_vs_leadtime(table, predictor="orog", score="rho2")
    plot_variance_explained_vs_leadtime(table, predictor="orog", score="pearson_log2")
    plot_variance_budget(metrics, table, var=HEADLINE_VAR, predictor="orog")
    plot_headline_scatter(metrics, predictors, var=HEADLINE_VAR, fhr=24, predictor="orog")

    # Honest CIs under spatial autocorrelation, for the t2m headline.
    logger.info(
        f"\n=== {HEADLINE_VAR}: spatial block-bootstrap 95% CI "
        f"(block={BLOCK} cells, {N_BOOT} reps) ==="
    )
    for metric in ["bias", "variance"]:
        for fhr in HEADLINE_FHRS:
            ci = block_bootstrap_ci(
                metrics[metric][HEADLINE_VAR].sel(fhr=fhr).values,
                predictors["orog"].values,
            )
            point = correlations(
                metrics[metric][HEADLINE_VAR].sel(fhr=fhr).values.ravel(),
                predictors["orog"].values.ravel(),
            )
            logger.info(
                f"{metric:8s} fhr={fhr:2d}  "
                f"Pearson {point['pearson_raw']:+.2f} "
                f"[{ci['pearson_ci'][0]:+.2f},{ci['pearson_ci'][1]:+.2f}]   "
                f"Spearman {point['spearman']:+.2f} "
                f"[{ci['spearman_ci'][0]:+.2f},{ci['spearman_ci'][1]:+.2f}]"
            )


if __name__ == "__main__":
    main()
