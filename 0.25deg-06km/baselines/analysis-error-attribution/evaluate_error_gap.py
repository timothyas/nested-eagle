"""
Attribute the per-cell error *gap* between two models to topography.

evaluate_topo_errors.py relates each model's RMSE to topography separately; the
gfs-only minus gfs-hrrr comparison there is indirect (two R^2 tables the reader
must eyeball).  This script instead forms the per-cell difference

    drmse     = rmse_gap - rmse_ref                         (units of the var)
    log_ratio = log10(rmse_gap / rmse_ref)                  (dimensionless)

at each station and regresses it directly on elevation and |grad h|.  If the
coarse model's extra error is terrain-driven, the gap should grow with
slope/elevation -- a single, direct test of the resolution hypothesis.

Both models verify against the same obs network and key each error on the
observation's own (rounded) location, so they share identical station ids: the
gap is an exact inner join on ``station``, with no regridding. Stations observed
by only one model, or outside the HRRR domain (topography ``valid`` mask), drop
out of the join.

Weighting.  A station's gap is estimated from N_ref and N_gap obs; the variance of
the difference goes as 1/N_ref + 1/N_gap, so we weight by the inverse of that,
w = N_ref*N_gap / (N_ref + N_gap) (the effective sample size of the difference).

Products (written to ./<gap>-vs-<ref>/ by default), one set per response:
  1. gap_map_<var>_<response>.png        -- station cells on lon/lat colored by
     the gap (diverging, centered at 0), fhr 0 vs 24; shows *where* the gap is.
  2. bivariate_hist_<var>_<response>.png -- gap vs each predictor per fhr, with
     the weighted OLS fit and a binned-median overlay.
  3. conditional_means_<response>.png    -- binned median gap +/- IQR vs each
     predictor, comparing fhr 0 vs 24.
  4. gap_regression_summary.csv          -- univariate + joint regression stats.
"""
import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# shared primitives + config, so the two scripts stay in lockstep
from evaluate_topo_errors import (
    MODEL_TAGS, VARS, FHRS, MIN_COUNT, GRAD_FLOOR_MKM,
    metrics_dir_for, weighted_linregress, weighted_multi, binned_stat,
)

# --- config ------------------------------------------------------------------
DEFAULT_GAP_MODEL = "gfs-only"    # the higher-error model (numerator of ratio)
DEFAULT_REF_MODEL = "gfs-hrrr"    # the lower-error reference (defines the grid)

# predictor key -> (column, axis label, is_log).  No log-elevation here: it was
# the weakest predictor in the single-model analysis (sea-level floor kills it).
PREDICTORS = {
    "orog_lin": ("orography", "elevation [m]", False),
    "grad_lin": ("grad_mkm", "|grad h| [m/km]", False),
    "grad_log": ("grad_log10", "log10 |grad h| [m/km]", True),
}

# response key -> column; log_ratio is primary (scale-free, symmetric)
RESPONSES = {"log_ratio": "log_ratio", "drmse": "drmse"}

# joint (multiple-regression) model -> predictor columns to combine
JOINT_MODELS = {
    "orog+grad": ["orography", "grad_mkm"],
    "orog+loggrad": ["orography", "grad_log10"],
}


def gap_label(var, rkey):
    """Y-axis / colorbar label for a (variable, response) pair."""
    name, unit = VARS[var]
    return f"Δ {name} RMSE [{unit}]" if rkey == "drmse" else f"log10 {name} RMSE ratio"


# --- data assembly -----------------------------------------------------------
def load_model(model, topo_file=None):
    """Open (rmse, count, topo) for a model from its default metrics dir."""
    tag = MODEL_TAGS[model]
    d = metrics_dir_for(model)
    rmse = xr.open_dataset(f"{d}/spatial.rmse.convobs.{tag}.nc")
    count = xr.open_dataset(f"{d}/spatial.count.convobs.{tag}.nc")
    topo = xr.open_dataset(topo_file or f"{d}/orography_and_gradient.nc")
    return rmse, count, topo


def build_gap_frame(var, fhr, rmse_gap, count_gap, rmse_ref, count_ref, topo):
    """Tidy per-station frame of the gap + predictors for one (var, fhr).

    The gap and reference datasets carry their own station axes; combining them
    here aligns on the shared station ids (union, then the finite/count filters
    below reduce it to the stations both models observed)."""
    valid = topo["valid"].astype(bool)
    ds = xr.merge(
        [
            rmse_gap[var].sel(fhr=fhr).rename("rmse_gap"),
            rmse_ref[var].sel(fhr=fhr).rename("rmse_ref"),
            count_gap[var].sel(fhr=fhr).rename("count_gap"),
            count_ref[var].sel(fhr=fhr).rename("count_ref"),
            topo["orography"],
            (topo["grad_mag"] * 1000.0).rename("grad_mkm"),  # m/m -> m/km
        ],
        join="inner",  # keep only stations present in both models (and topo)
        compat="override",  # overlapping coords (lat/lon) are identical
    ).where(valid)
    df = ds.to_dataframe().reset_index()
    df = df[(df["count_gap"] >= MIN_COUNT) & (df["count_ref"] >= MIN_COUNT)
            & np.isfinite(df["rmse_gap"]) & np.isfinite(df["rmse_ref"])
            & (df["rmse_gap"] > 0) & (df["rmse_ref"] > 0)].copy()
    df["grad_log10"] = np.log10(df["grad_mkm"].clip(lower=GRAD_FLOOR_MKM))
    df["drmse"] = df["rmse_gap"] - df["rmse_ref"]
    df["log_ratio"] = np.log10(df["rmse_gap"]) - np.log10(df["rmse_ref"])
    # inverse-variance weight for a difference of two cell means
    df["w"] = df["count_gap"] * df["count_ref"] / (df["count_gap"] + df["count_ref"])
    return df


# --- regression --------------------------------------------------------------
def regression_rows(df, var, fhr, rkey):
    """Univariate + joint regression stats for one (var, fhr, response)."""
    y = df[RESPONSES[rkey]].values
    w = df["w"].values
    rows = []
    for pk, (col, _, _) in PREDICTORS.items():
        x = df[col].values
        wl = weighted_linregress(x, y, w)
        pr = stats.pearsonr(x, y)
        sp = stats.spearmanr(x, y)              # transform-invariant in y
        uni = weighted_linregress(x, y, np.ones_like(w))  # unweighted
        rows.append(dict(
            response=rkey, variable=var, fhr=fhr, predictor=pk, n=len(df),
            slope_w=wl["slope"], intercept_w=wl["intercept"],
            r2_w=wl["r2"], r2_unweighted=uni["r2"],
            pearson_r=pr.statistic, pearson_p=pr.pvalue,
            spearman_r=sp.statistic, spearman_p=sp.pvalue,
        ))
    for tag, cols in JOINT_MODELS.items():
        r2j, betas = weighted_multi([df[c].values for c in cols], y, w)
        row = dict(response=rkey, variable=var, fhr=fhr, predictor=tag,
                   n=len(df), r2_w=r2j)
        row.update({f"beta_{c}": b for c, b in zip(cols, betas)})
        rows.append(row)
    return rows


# --- plots -------------------------------------------------------------------
def gap_map_figure(frames, var, rkey, out_dir):
    """Station cells on lon/lat colored by the gap, fhr 0 vs 24."""
    ycol = RESPONSES[rkey]
    clabel = gap_label(var, rkey)
    # symmetric color range from a robust percentile across both fhrs
    allv = np.concatenate([frames[f][ycol].values for f in FHRS])
    vmax = np.nanpercentile(np.abs(allv), 98)
    fig, axes = plt.subplots(1, len(FHRS), figsize=(6.2 * len(FHRS), 4.6),
                             constrained_layout=True, sharex=True, sharey=True)
    for ax, fhr in zip(np.atleast_1d(axes), FHRS):
        df = frames[fhr]
        sc = ax.scatter(df["longitude"], df["latitude"], c=df[ycol], s=8,
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"fhr={fhr}")
        ax.set_xlabel("longitude")
        ax.set_aspect("equal", adjustable="box")
    np.atleast_1d(axes)[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes, shrink=0.85, label=clabel)
    fig.suptitle(f"{var}: error gap (red = {DEFAULT_GAP_MODEL} worse)", fontsize=13)
    path = os.path.join(out_dir, f"gap_map_{var}_{rkey}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def bivariate_hist_figure(frames, var, rkey, out_dir):
    ycol = RESPONSES[rkey]
    ylabel = gap_label(var, rkey)
    keys = list(PREDICTORS)
    fig, axes = plt.subplots(len(keys), len(FHRS), figsize=(11, 3.5 * len(keys)),
                             constrained_layout=True)
    for i, pk in enumerate(keys):
        col, xlabel, _ = PREDICTORS[pk]
        for j, fhr in enumerate(FHRS):
            ax = axes[i, j]
            df = frames[fhr]
            sns.histplot(df, x=col, y=ycol, bins=40, cmap="mako",
                         cbar=(j == len(FHRS) - 1), ax=ax)
            x, y, w = df[col].values, df[ycol].values, df["w"].values
            fit = weighted_linregress(x, y, w)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, fit["intercept"] + fit["slope"] * xs, "r-", lw=2,
                    label=f"OLS r$^2$={fit['r2']:.2f}")
            bs = binned_stat(x, y)
            if bs.size:
                ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                        label="binned median")
            ax.axhline(0.0, color="0.5", lw=1, ls="--")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_title(f"fhr={fhr}")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    fig.suptitle(f"{var}: error gap vs topography (n={len(frames[FHRS[0]])} cells, "
                 f"weighted fit)", fontsize=13)
    path = os.path.join(out_dir, f"bivariate_hist_{var}_{rkey}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def conditional_means_figure(all_frames, rkey, out_dir):
    ycol = RESPONSES[rkey]
    keys = list(PREDICTORS)
    fig, axes = plt.subplots(len(keys), len(VARS), figsize=(11, 3.5 * len(keys)),
                             constrained_layout=True)
    for j, var in enumerate(VARS):
        ylabel = gap_label(var, rkey)
        for i, pk in enumerate(keys):
            col, xlabel, _ = PREDICTORS[pk]
            ax = axes[i, j]
            for fhr in FHRS:
                df = all_frames[(var, fhr)]
                bs = binned_stat(df[col].values, df[ycol].values, nbins=18)
                if not bs.size:
                    continue
                line, = ax.plot(bs[0], bs[1], "o-", ms=4, label=f"fhr={fhr}")
                ax.fill_between(bs[0], bs[2], bs[3], alpha=0.15,
                                color=line.get_color())
            ax.axhline(0.0, color="0.5", lw=1, ls="--")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if j == 0 else "")
            if i == 0:
                ax.set_title(var)
            ax.legend(fontsize=8)
    fig.suptitle("Binned median error gap +/- IQR vs topography", fontsize=13)
    path = os.path.join(out_dir, f"conditional_means_{rkey}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- main --------------------------------------------------------------------
def main(gap_model, ref_model, out_dir, topo_file=None):
    os.makedirs(out_dir, exist_ok=True)
    rmse_ref, count_ref, topo = load_model(ref_model, topo_file)  # stations + topography
    rmse_gap, count_gap, _ = load_model(gap_model)

    all_frames = {(var, fhr): build_gap_frame(var, fhr, rmse_gap, count_gap,
                                              rmse_ref, count_ref, topo)
                  for var in VARS for fhr in FHRS}

    # headline: how big is the gap, and does it track slope?
    print(f"Error gap  {gap_model}  minus  {ref_model}  (positive = "
          f"{gap_model} worse):")
    for (var, fhr), df in all_frames.items():
        med = np.median(df["log_ratio"])
        frac = np.mean(df["log_ratio"] > 0)
        r = stats.spearmanr(df["grad_mkm"], df["log_ratio"]).statistic
        print(f"  {var:16s} fhr={fhr:2d}: median log-ratio={med:+.3f} "
              f"({100 * frac:4.1f}% of cells worse)  rho(log-ratio, slope)={r:+.2f}")

    rows = []
    for rkey in RESPONSES:
        for var in VARS:
            for fhr in FHRS:
                rows += regression_rows(all_frames[(var, fhr)], var, fhr, rkey)

    summary = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "gap_regression_summary.csv")
    summary.to_csv(csv, index=False)

    print("\n=== Univariate weighted fits (R^2) ===")
    uni = summary[summary["predictor"].isin(PREDICTORS)]
    pivot = uni.pivot_table(index=["response", "variable", "fhr"],
                            columns="predictor", values="r2_w")
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== Joint regression (standardized betas, weighted R^2) ===")
    joint = summary[summary["predictor"].isin(JOINT_MODELS)]
    beta_cols = [c for c in summary.columns if c.startswith("beta_")]
    for _, r in joint.iterrows():
        betas = "  ".join(f"{c[len('beta_'):]}={r[c]:+.2f}"
                          for c in beta_cols if pd.notna(r[c]))
        print(f"  {r['response']:9s} {r['variable']:16s} fhr={int(r['fhr']):2d} "
              f"{r['predictor']:14s} R2={r['r2_w']:.3f}  {betas}")

    paths = []
    for rkey in RESPONSES:
        paths.append(conditional_means_figure(all_frames, rkey, out_dir))
        for var in VARS:
            frames = {fhr: all_frames[(var, fhr)] for fhr in FHRS}
            paths.append(gap_map_figure(frames, var, rkey, out_dir))
            paths.append(bivariate_hist_figure(frames, var, rkey, out_dir))
    print("\nWrote:")
    for p in [csv] + paths:
        print(" ", p)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gap-model", choices=list(MODEL_TAGS), default=DEFAULT_GAP_MODEL,
                   help="higher-error model, regridded onto the reference "
                        "(default: %(default)s)")
    p.add_argument("--ref-model", choices=list(MODEL_TAGS), default=DEFAULT_REF_MODEL,
                   help="reference model; its station topography file is used "
                        "(default: %(default)s)")
    p.add_argument("--topo-file", default=None,
                   help="station orography_and_gradient.nc (default: ref model's); "
                        "point at an alternate-DEM file to compare orographies")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: ./<gap>-vs-<ref>)")
    args = p.parse_args()
    out_dir = args.out_dir or f"./{args.gap_model}-vs-{args.ref_model}"
    main(args.gap_model, args.ref_model, out_dir, args.topo_file)
