"""
Relate spatial RMSE (against conventional surface obs) to topography.

For 2m_temperature and 10m_wind_speed at fhr 0 and 24, we ask how much of the
per-cell error pattern is explained by:
  - orography   : the (regridded) surface elevation, and
  - grad_mag    : the amplitude of the local topographic gradient |grad h|.

We evaluate the error two ways (the RESPONSES below): as raw RMSE and as
log10(RMSE).  Per-cell RMSE is heavy-tailed and its log is far closer to
Gaussian, so the log response both satisfies the OLS assumptions better and
recasts the fit as a *multiplicative* statement (a fractional change in RMSE
per unit predictor).  Count-weighting is also better justified there: the
sampling variance of log(RMSE) for a cell with N obs is ~1/(2N), so weighting
by obs count approximates inverse-variance weighting.

Conventional surface obs are spatially sparse, so per-cell RMSE only exists at
the ~5k cells that contain a station; those cells are well sampled though
(median ~450 obs each). We keep cells that are inside the HRRR domain
(``valid``) and backed by at least MIN_COUNT obs.

Each predictor is taken linearly and (for the strictly-positive, right-skewed
gradient) as log10.  Elevation is also offered as log10, but note it has a true
zero at sea level and can be negative (coastal / below-sea-level cells), so the
log form is floored at ELEV_FLOOR_M and should be read with that caveat -- it is
less physically motivated than log-gradient.

Products (written to the output dir, ./<model>/ by default), one set per
response (raw RMSE and log10 RMSE):
  1. bivariate_hist_<var>_<response>.png -- seaborn 2-D histograms of the
     response vs each predictor per fhr, with the count-weighted OLS fit and a
     binned-median overlay.
  2. conditional_means_<response>.png    -- binned median response +/- IQR vs
     each predictor, a robust view comparing fhr 0 vs 24.
  3. regression_summary.csv              -- univariate (Pearson/Spearman/
     weighted OLS) and multiple-regression statistics, for both responses,
     weighted and unweighted.  (Spearman is transform-invariant, so it repeats
     across responses by construction.)
"""
import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- config ------------------------------------------------------------------
SCRATCH = os.environ["SCRATCH"]
# forecast source -> filename tag of its spatial-obs-metrics files
MODEL_TAGS = {"gfs-hrrr": "nested-lam", "gfs-only": "global"}
DEFAULT_MODEL = "gfs-hrrr"


def metrics_dir_for(model):
    """Default location of the spatial-obs-metrics files for a given model."""
    return (
        f"{SCRATCH}/nested-eagle/0.25deg-06km/production/{model}/stage1c"
        f"/inference-testing/spatial-obs-metrics"
    )

# variable -> (short name, unit) used to build response axis labels
VARS = {"2m_temperature": ("2m T", "K"), "10m_wind_speed": ("10m wind", "m/s")}
FHRS = [0, 24]
MIN_COUNT = 30          # drop cells whose RMSE rests on too few obs
GRAD_FLOOR_MKM = 0.1    # floor (m/km) for log10(grad) on truly-flat cells
ELEV_FLOOR_M = 1.0      # floor (m) for log10(elevation); see caveat in build_frame

# predictor key -> (column, axis label, is_log)
PREDICTORS = {
    "orog_lin": ("orography", "elevation [m]", False),
    "orog_log": ("orog_log10", "log10 elevation [m]", True),
    "grad_lin": ("grad_mkm", "|grad h| [m/km]", False),
    "grad_log": ("grad_log10", "log10 |grad h| [m/km]", True),
}

# response key -> (column, short label, is_log)
RESPONSES = {
    "rmse": ("rmse", "RMSE", False),
    "log_rmse": ("log10_rmse", "log10 RMSE", True),
}

# joint (multiple-regression) model -> predictor columns to combine
JOINT_MODELS = {
    "orog+grad": ["orography", "grad_mkm"],
    "orog+loggrad": ["orography", "grad_log10"],
    "logorog+loggrad": ["orog_log10", "grad_log10"],
}


def response_label(var, rkey):
    """Y-axis label for a (variable, response) pair, e.g. '2m T RMSE [K]'."""
    name, unit = VARS[var]
    return f"log10 {name} RMSE" if RESPONSES[rkey][2] else f"{name} RMSE [{unit}]"


# --- weighted statistics helpers --------------------------------------------
def w_moments(x, w):
    W = w.sum()
    m = np.sum(w * x) / W
    v = np.sum(w * (x - m) ** 2) / W
    return m, v


def weighted_linregress(x, y, w):
    """Count-weighted simple OLS y ~ a + b x; returns slope, intercept, r, r2."""
    mx, vx = w_moments(x, w)
    my, vy = w_moments(y, w)
    cxy = np.sum(w * (x - mx) * (y - my)) / w.sum()
    b = cxy / vx
    r = cxy / np.sqrt(vx * vy)
    return dict(slope=b, intercept=my - b * mx, r=r, r2=r * r)


def weighted_multi(cols, y, w):
    """Weighted multiple regression. Returns weighted R2 and standardized betas.

    Standardizing each predictor and the response by their weighted std makes the
    betas directly comparable despite orography/grad being on different scales
    (and correlated)."""
    yz_m, yz_v = w_moments(y, w)
    ys = (y - yz_m) / np.sqrt(yz_v)
    Z = []
    for x in cols:
        m, v = w_moments(x, w)
        Z.append((x - m) / np.sqrt(v))
    X = np.column_stack([np.ones(len(y))] + Z)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], ys * sw, rcond=None)
    yhat = X @ coef
    ss_res = np.sum(w * (ys - yhat) ** 2)
    ss_tot = np.sum(w * (ys - 0.0) ** 2)  # ys already weighted-centered
    return 1 - ss_res / ss_tot, coef[1:]


# --- data assembly -----------------------------------------------------------
def build_frame(var, fhr, rmse, count, topo):
    """Tidy DataFrame of per-cell RMSE + predictors for one (var, fhr)."""
    valid = topo["valid"].astype(bool)
    # inner merge on station: topo may be a superset (one HRRR-orography file
    # shared across models), so keep only this model's stations.
    ds = xr.merge(
        [
            rmse[var].sel(fhr=fhr).rename("rmse"),
            count[var].sel(fhr=fhr).rename("count"),
            topo["orography"],
            (topo["grad_mag"] * 1000.0).rename("grad_mkm"),  # m/m -> m/km
        ],
        join="inner", compat="override",  # overlapping coords (lat/lon) are identical
    ).where(valid)
    df = ds.to_dataframe().reset_index()
    df = df[(df["count"] >= MIN_COUNT) & np.isfinite(df["rmse"])
            & (df["rmse"] > 0)].copy()  # rmse > 0 so log10_rmse is finite
    df["grad_log10"] = np.log10(df["grad_mkm"].clip(lower=GRAD_FLOOR_MKM))
    # Elevation has a true zero (sea level) and goes negative at coastal /
    # below-sea-level cells, so floor before logging; the low end is distorted.
    df["orog_log10"] = np.log10(df["orography"].clip(lower=ELEV_FLOOR_M))
    df["log10_rmse"] = np.log10(df["rmse"])
    return df


# --- plots -------------------------------------------------------------------
def binned_stat(x, y, nbins=15):
    """Quantile-binned median (and IQR) of y vs x, returned at bin x-medians."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        out.append((np.median(x[m]), np.median(y[m]),
                    np.quantile(y[m], 0.25), np.quantile(y[m], 0.75)))
    return np.array(out).T if out else np.empty((4, 0))


def bivariate_hist_figure(frames, var, rkey, out_dir):
    ycol, rlabel, _ = RESPONSES[rkey]
    ylabel = response_label(var, rkey)
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
            x, y, w = df[col].values, df[ycol].values, df["count"].values
            fit = weighted_linregress(x, y, w)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, fit["intercept"] + fit["slope"] * xs, "r-", lw=2,
                    label=f"OLS r$^2$={fit['r2']:.2f}")
            bs = binned_stat(x, y)
            if bs.size:
                ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                        label="binned median")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_title(f"fhr={fhr}")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
    fig.suptitle(f"{var}: {rlabel} vs topography (n={len(frames[FHRS[0]])} cells, "
                 f"count-weighted fit)", fontsize=13)
    path = os.path.join(out_dir, f"bivariate_hist_{var}_{rkey}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def conditional_means_figure(all_frames, rkey, out_dir):
    ycol, rlabel, _ = RESPONSES[rkey]
    keys = list(PREDICTORS)
    fig, axes = plt.subplots(len(keys), len(VARS), figsize=(11, 3.5 * len(keys)),
                             constrained_layout=True)
    for j, var in enumerate(VARS):
        ylabel = response_label(var, rkey)
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
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel if j == 0 else "")
            if i == 0:
                ax.set_title(var)
            ax.legend(fontsize=8)
    fig.suptitle(f"Binned median {rlabel} +/- IQR vs topography", fontsize=13)
    path = os.path.join(out_dir, f"conditional_means_{rkey}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- regression --------------------------------------------------------------
def regression_rows(df, var, fhr, rkey):
    """Univariate + joint regression stats for one (var, fhr, response)."""
    y = df[RESPONSES[rkey][0]].values
    w = df["count"].values
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


# --- main --------------------------------------------------------------------
def main(metrics_dir, model_tag, out_dir, topo_file):
    os.makedirs(out_dir, exist_ok=True)
    rmse = xr.open_dataset(f"{metrics_dir}/spatial.rmse.convobs.{model_tag}.nc")
    count = xr.open_dataset(f"{metrics_dir}/spatial.count.convobs.{model_tag}.nc")
    topo = xr.open_dataset(topo_file)

    all_frames = {(var, fhr): build_frame(var, fhr, rmse, count, topo)
                  for var in VARS for fhr in FHRS}

    # collinearity of the predictors entering the joint models (with two
    # predictors, the VIF is just 1/(1 - r^2), so the pairwise r tells the story)
    print("Predictor collinearity (Pearson r), per (var, fhr):")
    for (var, fhr), df in all_frames.items():
        r_og = stats.pearsonr(df["orography"], df["grad_mkm"])[0]
        r_lg = stats.pearsonr(df["orography"], df["grad_log10"])[0]
        r_ll = stats.pearsonr(df["orog_log10"], df["grad_log10"])[0]
        print(f"  {var:16s} fhr={fhr:2d}: corr(orog,grad)={r_og:+.2f} "
              f"corr(orog,log grad)={r_lg:+.2f} "
              f"corr(log orog,log grad)={r_ll:+.2f}")

    rows = []
    for rkey in RESPONSES:
        for var in VARS:
            for fhr in FHRS:
                rows += regression_rows(all_frames[(var, fhr)], var, fhr, rkey)

    summary = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "regression_summary.csv")
    summary.to_csv(csv, index=False)

    print("\n=== Univariate count-weighted fits (R^2) ===")
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
              f"{r['predictor']:16s} R2={r['r2_w']:.3f}  {betas}")

    paths = []
    for rkey in RESPONSES:
        paths.append(conditional_means_figure(all_frames, rkey, out_dir))
        for var in VARS:
            frames = {fhr: all_frames[(var, fhr)] for fhr in FHRS}
            paths.append(bivariate_hist_figure(frames, var, rkey, out_dir))
    print("\nWrote:")
    for p in [csv] + paths:
        print(" ", p)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=list(MODEL_TAGS), default=DEFAULT_MODEL,
                   help="forecast source to evaluate (default: %(default)s)")
    p.add_argument("--metrics-dir", default=None,
                   help="dir holding spatial.*.convobs.<tag>.nc (default: derived from --model)")
    p.add_argument("--topo-file", default=None,
                   help="station orography_and_gradient.nc; one HRRR-orography file "
                        "can be shared across models (default: <metrics-dir>/orography_and_gradient.nc)")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: ./<model>)")
    args = p.parse_args()
    metrics_dir = args.metrics_dir or metrics_dir_for(args.model)
    topo_file = args.topo_file or f"{metrics_dir}/orography_and_gradient.nc"
    out_dir = args.out_dir or f"./{args.model}"
    main(metrics_dir, MODEL_TAGS[args.model], out_dir, topo_file)
