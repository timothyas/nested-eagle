"""
CONUS maps that place the 2 m T analysis bias (HRRR - GFS) next to the terrain
difference the two models resolve, to show the bias is an elevation artifact.

Three panels, all at the obs-station locations:
  1. HRRR - GFS elevation (orog_diff): where HRRR sits lower/higher than GFS.
  2. HRRR - GFS 2 m T bias: the analysis temperature difference.
  3. Residual bias after removing the fitted lapse-rate term
     (bias - slope * orog_diff): what is left once the elevation difference is
     accounted for. The collapse in spread is the attribution.

The first two panels share a sign-flipped relationship (lower HRRR terrain ->
warmer HRRR), so panel 1 uses a reversed diverging map to make the spatial
correspondence read as "same colour = consistent with the lapse rate".

Output: terrain_maps_gfs_vs_hrrr.png
"""
import math
import os

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

DATA = f"{os.environ['SCRATCH']}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"
OUT = "terrain_maps_gfs_vs_hrrr.png"
EXTENT = [-125, -66, 23, 50]  # CONUS


def load():
    m = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.metrics.nc").sel(field="2m_temperature")
    t = xr.open_dataset(f"{DATA}/gfs_vs_hrrr.topo.nc")
    lon = ((t["orog_diff"]["longitude"].values + 180) % 360) - 180  # 0..360 -> -180..180
    lat = t["orog_diff"]["latitude"].values
    return lon, lat, t["orog_diff"].values, m["bias"].values


def nice_step(vlim):
    """A round step so 3.5*step (the outer bin edge) sits just past vlim."""
    raw = vlim / 3.5
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def discrete_bins(vlim):
    """Symmetric 7-bin boundaries: 3 negative / 1 near-zero / 3 positive.

    The central bin straddles zero (+/-0.5 step), so near-zero stations share one
    neutral colour; values beyond the outer edges fall into the extend triangles.
    """
    s = nice_step(vlim)
    return s * np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])


def scatter_map(ax, lon, lat, c, vlim, cmap_name, title, cbar_label, fmt="%g"):
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.6")
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    bounds = discrete_bins(vlim)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, extend="both")
    sc = ax.scatter(lon, lat, c=c, s=9, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(), edgecolor="none")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9,
                      extend="both", ticks=bounds, spacing="uniform",
                      format=fmt)
    cb.set_label(cbar_label, fontsize=9)


def main():
    lon, lat, orog_diff, bias = load()
    fit = stats.linregress(orog_diff, bias)
    resid = bias - (fit.intercept + fit.slope * orog_diff)
    var_expl = 1 - np.nanvar(resid) / np.nanvar(bias)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    # Panel 1: terrain difference, reversed map so "warm-ish" colour = lower HRRR
    # terrain (which drives a warm HRRR bias), aligning visually with panel 2.
    o_lim = np.nanpercentile(np.abs(orog_diff), 98)
    # RdBu => negative (HRRR lower) renders red, matching the warm (red) T bias
    # in panel 2, so a lapse-consistent station is the same colour in both.
    scatter_map(axes[0], lon, lat, orog_diff, o_lim, "RdBu",
                "HRRR - GFS elevation", "elevation diff [m]  (red = HRRR lower)",
                fmt="%.0f")
    b_lim = np.nanpercentile(np.abs(bias), 98)
    scatter_map(axes[1], lon, lat, bias, b_lim, "RdBu_r",
                "HRRR - GFS 2 m T bias", "T bias [K]  (red = HRRR warmer)",
                fmt="%.1f")
    scatter_map(axes[2], lon, lat, resid, b_lim, "RdBu_r",
                f"residual after lapse-rate removal\n"
                f"({fit.slope*1000:.2f} K/km; {var_expl*100:.0f}% of bias variance removed)",
                "residual T bias [K]", fmt="%.1f")
    fig.suptitle("2 m T analysis bias tracks the resolved-terrain difference "
                 f"(n={len(bias)} stations, HRRR - GFS)", fontsize=14)
    fig.savefig(OUT, dpi=130)
    print(f"Wrote {OUT}")
    print(f"fitted lapse rate {-fit.slope*1000:.2f} K/km, r={fit.rvalue:+.2f}; "
          f"bias std {np.nanstd(bias):.2f} K -> residual std {np.nanstd(resid):.2f} K "
          f"({var_expl*100:.0f}% of variance removed)")


if __name__ == "__main__":
    main()
