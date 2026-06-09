"""
Variance side of the GFS<->HRRR analysis comparison, against terrain.

Two questions, two figures (all at obs stations, HRRR - GFS convention):

1. variance_terrain_gfs_vs_hrrr.png
   Does the *error variance* var_t(HRRR - GFS) -- the time-varying part of the
   disagreement, which is the dominant term in MSE (~83% for 2 m T) -- grow where
   the two models' terrain disagrees? var_diff vs |orog_diff| and |slope_diff|.
   This is a genuine but modest signal (weaker than the mean-bias lapse-rate one).

2. variance_ratio_maps_gfs_vs_hrrr.png
   Does HRRR resolve *more* temporal variability than GFS, especially in complex
   terrain? Map log2(var_hrrr / var_gfs). The answer is essentially no: the ratio
   hovers at ~1 everywhere (pale near-zero band) and does not track terrain -- a
   null worth showing.
"""
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

from plot_terrain_relation import binned_stat
from plot_terrain_maps import discrete_bins, EXTENT

import matplotlib as mpl

DATA = f"{os.environ['SCRATCH']}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"
FIELDS = {"2m_temperature": ("2 m T", "K"), "10m_wind_speed": ("10 m wind", "m/s")}


def load():
    m = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.metrics.nc")
    t = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.topo.nc")
    lon = ((t["orog_diff"]["longitude"].values + 180) % 360) - 180
    lat = t["orog_diff"]["latitude"].values
    return m, t, lon, lat


def relation_panel(ax, x, y, xlabel, ylabel):
    sns_ok = True
    try:
        import seaborn as sns
        sns.histplot(x=x, y=y, bins=45, cmap="mako", ax=ax, cbar=False)
    except Exception:
        sns_ok = False
        ax.scatter(x, y, s=4, alpha=0.3, color="0.4")
    lin = stats.linregress(x, y)
    rho = stats.spearmanr(x, y).statistic
    xs = np.array([np.quantile(x, 0.005), np.quantile(x, 0.995)])
    ax.plot(xs, lin.intercept + lin.slope * xs, "r-", lw=2,
            label=f"OLS r={lin.rvalue:+.2f}  $\\rho$={rho:+.2f}")
    bs = binned_stat(x, y)
    if bs.size:
        ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                label="binned median")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8, framealpha=0.75)


def variance_terrain_figure(m, t):
    od = np.abs(t["orog_diff"].values)
    sd = np.abs(t["slope_diff"].values) * 1000.0  # m/m -> m/km
    fig, axes = plt.subplots(len(FIELDS), 2, figsize=(11, 4.2 * len(FIELDS)),
                             constrained_layout=True)
    for i, (f, (lab, unit)) in enumerate(FIELDS.items()):
        vd = m["var_diff"].sel(field=f).values
        g = np.isfinite(vd)
        relation_panel(axes[i, 0], od[g], vd[g],
                       "|HRRR - GFS elevation| [m]", f"{lab} var(HRRR-GFS) [{unit}$^2$]")
        relation_panel(axes[i, 1], sd[g], vd[g],
                       "|HRRR - GFS slope| [m/km]", f"{lab} var(HRRR-GFS) [{unit}$^2$]")
    fig.suptitle("Error variance var(HRRR - GFS) vs terrain mismatch "
                 f"(n={int(np.isfinite(vd).sum())} stations)", fontsize=13)
    out = "variance_terrain_gfs_vs_hrrr.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def ratio_map(ax, lon, lat, ratio, title):
    logr = np.log2(ratio)
    vlim = np.nanpercentile(np.abs(logr), 98)
    bounds = discrete_bins(vlim)
    cmap = plt.get_cmap("RdBu_r")  # >1 (HRRR more variance) -> red
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend="both")
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.6")
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    sc = ax.scatter(lon, lat, c=logr, s=9, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(), edgecolor="none")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9,
                      extend="both", ticks=bounds, spacing="uniform")
    # label ticks as the underlying variance ratio (2**log2ratio), not the log
    cb.ax.set_xticklabels([f"{2**b:.2f}" for b in bounds], fontsize=7)
    cb.set_label("var_hrrr / var_gfs  (red = HRRR more variable)", fontsize=9)


def variance_ratio_figure(m, lon, lat):
    fig, axes = plt.subplots(1, len(FIELDS), figsize=(13, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    for ax, (f, (lab, _)) in zip(axes, FIELDS.items()):
        s = m.sel(field=f)
        ratio = s["var_hrrr"].values / s["var_gfs"].values
        med = np.nanmedian(ratio)
        ratio_map(ax, lon, lat, ratio, f"{lab}: var_hrrr / var_gfs  (median {med:.2f})")
    fig.suptitle("HRRR temporal variance is ~equal to GFS and does not track "
                 "terrain (ratio ~ 1 everywhere)", fontsize=13)
    out = "variance_ratio_maps_gfs_vs_hrrr.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def main():
    m, t, lon, lat = load()
    variance_terrain_figure(m, t)
    variance_ratio_figure(m, lon, lat)


if __name__ == "__main__":
    main()
