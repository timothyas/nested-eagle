"""
Surface-roughness explanation of the 10 m wind difference (HRRR - GFS).

Using the true roughness difference rough_diff = log10(z0_hrrr) - log10(z0_gfs)
pulled from GFS/HRRR GRIB SFCR (see fetch_roughness.py), we show the wind bias
tracks roughness with the physical sign (HRRR rougher -> slower -> negative bias)
and -- unlike the 2 m T vs elevation case -- robustly (Pearson ~ Spearman, so it
is not a leverage artifact).

Figures:
  roughness_relation_gfs_vs_hrrr.png : wind bias vs rough_diff (log-count density,
      Theil-Sen + Spearman), and var_diff vs |rough_diff|.
  roughness_maps_gfs_vs_hrrr.png     : CONUS rough_diff next to wind bias, with
      matched colours (HRRR rougher = HRRR slower = same colour).
"""
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from scipy import stats

from plot_terrain_relation import binned_stat
from plot_terrain_maps import scatter_map, EXTENT

DATA = f"{os.environ['SCRATCH']}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"


def load():
    m = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.metrics.nc").sel(field="10m_wind_speed")
    r = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.roughness.nc")
    lon = ((r["rough_diff"]["longitude"].values + 180) % 360) - 180
    lat = r["rough_diff"]["latitude"].values
    return m, r, lon, lat


def loghist_panel(ax, x, y, xlabel, ylabel):
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    xr_ = np.percentile(x, [0.5, 99.5])
    yr_ = np.percentile(y, [0.5, 99.5])
    h = ax.hist2d(x, y, bins=45, range=[xr_, yr_], norm=LogNorm(), cmap="mako")
    cb = plt.colorbar(h[3], ax=ax, pad=0.01, fraction=0.046)
    cb.set_label("bin count (log)", fontsize=8)
    ols = stats.linregress(x, y)
    ts = stats.theilslopes(y, x)
    rho = stats.spearmanr(x, y).statistic
    xs = np.array(xr_)
    ax.plot(xs, ols.intercept + ols.slope * xs, "-", color="red", lw=1.6,
            label=f"OLS  r={ols.rvalue:+.2f}")
    ax.plot(xs, ts[1] + ts[0] * xs, "--", color="lime", lw=1.8,
            label=f"Theil-Sen  $\\rho$={rho:+.2f}")
    bs = binned_stat(x, y)
    if bs.size:
        ax.plot(bs[0], bs[1], "o-", color="orange", ms=4, lw=1.5,
                label="binned median")
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_xlim(xr_)
    ax.set_ylim(yr_)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8, framealpha=0.75)


def relation_figure(m, r):
    rd = r["rough_diff"].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    loghist_panel(axes[0], rd, m["bias"].values,
                  "log10(z0 HRRR / z0 GFS)   (>0 = HRRR rougher)",
                  "10 m wind bias [m/s]")
    axes[0].set_title("HRRR rougher -> HRRR slower (negative bias)", fontsize=10)
    loghist_panel(axes[1], np.abs(rd), m["var_diff"].values,
                  "|log10(z0 HRRR / z0 GFS)|", "10 m wind var(HRRR-GFS) [m/s$^2$]")
    fig.suptitle("10 m wind difference vs surface-roughness difference "
                 f"(n={np.isfinite(rd).sum()} stations)", fontsize=13)
    out = "roughness_relation_gfs_vs_hrrr.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def maps_figure(m, r, lon, lat):
    rd = r["rough_diff"].values
    bias = m["bias"].values
    # Single OLS roughness fit, same construction as the 2 m T lapse-removal panel.
    # Roughness explains only a modest share here (R^2~0.18), so the residual panel
    # is meant to read that way: the bulk pattern survives, unlike the T case.
    g = np.isfinite(rd) & np.isfinite(bias)
    fit = stats.linregress(rd[g], bias[g])
    resid = bias - (fit.intercept + fit.slope * rd)
    var_expl = 1 - np.nanvar(resid[g]) / np.nanvar(bias[g])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    # RdBu => positive (HRRR rougher) renders blue, matching the slow (blue) wind
    # bias in panel 2: a roughness-consistent station is the same colour in both.
    # Scale to the 90th pct (not 98th): a few coast/water stations hit the z0
    # floor and give |rough_diff|~2-4, which would otherwise wash out the bulk.
    r_lim = np.nanpercentile(np.abs(rd), 90)
    scatter_map(axes[0], lon, lat, rd, r_lim, "RdBu",
                "HRRR - GFS log10 roughness",
                "log10(z0_hrrr/z0_gfs)  (blue = HRRR rougher)", fmt="%.2f")
    b_lim = np.nanpercentile(np.abs(bias), 98)
    scatter_map(axes[1], lon, lat, bias, b_lim, "RdBu_r",
                "HRRR - GFS 10 m wind bias",
                "wind bias [m/s]  (blue = HRRR slower)", fmt="%.1f")
    scatter_map(axes[2], lon, lat, resid, b_lim, "RdBu_r",
                f"residual after roughness removal\n"
                f"(slope {fit.slope:+.2f} m/s per log10 z0; "
                f"{var_expl*100:.0f}% of bias variance removed)",
                "residual wind bias [m/s]", fmt="%.1f")
    fig.suptitle("Where HRRR is rougher than GFS, its 10 m wind is slower "
                 f"(n={int(g.sum())} stations, HRRR - GFS; matched blue)",
                 fontsize=13)
    out = "roughness_maps_gfs_vs_hrrr.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")
    print(f"roughness slope {fit.slope:+.3f} m/s per log10 z0, r={fit.rvalue:+.2f}; "
          f"bias std {np.nanstd(bias[g]):.2f} -> residual std {np.nanstd(resid[g]):.2f} m/s "
          f"({var_expl*100:.0f}% of variance removed)")


def main():
    m, r, lon, lat = load()
    relation_figure(m, r)
    maps_figure(m, r, lon, lat)


if __name__ == "__main__":
    main()
