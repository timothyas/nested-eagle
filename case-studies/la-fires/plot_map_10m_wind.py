"""
2×2 map of 10 m wind speed over Southern California for a single valid time,
comparing Nested-EAGLE (LAM), GFS, HRRR, and Global-EAGLE.

PlateCarree projection. Each panel shows model 10 m wind speed (cmo.speed_r)
with light orography contours, overlaid with the case-study observation
stations (tier-colored, via plot_station_map.draw_stations) and the
Palisades/Eaton fire ignition points. Panel order matches the time-series
figure: Nested-EAGLE | GFS / HRRR | Global-EAGLE.

HRRR is only available out to 48 h; longer leads leave that panel empty.

Edit T0 and FHR (or set the MAP_T0 / MAP_FHR env vars) to change the forecast.
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from plot_station_map import draw_stations

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

# ── Parameters ────────────────────────────────────────────────────────────────
T0  = os.environ.get("MAP_T0", "2025-01-04T12")
FHR = int(os.environ.get("MAP_FHR", "96"))

VALID_TIME = pd.Timestamp(T0) + pd.Timedelta(hours=FHR)
SP_LEVELS   = np.arange(972, 1024, 4)          # surface pressure contour levels (hPa)
WSPD_VMAX   = 10.0                             # m/s, shared colorscale
OROG_LEVELS = [200, 500, 1000, 1500, 2000, 2500]  # orography contour levels (m)

# Map extent — wider than the station map to give the wind field more context
# (a tight crop makes the coarse GFS/Global-EAGLE cells dominate the panel).
_WEST  = -119.5
_EAST  = -117.0
_NORTH =   34.8
_SOUTH =   33.3

# ── Data paths ────────────────────────────────────────────────────────────────
EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")


# ── Data loading ──────────────────────────────────────────────────────────────

def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


def _select(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=VALID_TIME).squeeze()


def load_nested_eagle() -> xr.Dataset:
    path = os.path.join(EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="nested-lam",
        vars_of_interest=["u10", "v10", "sp"],
        load=True,
        reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480},
        lam_index=407040,
    )
    return _select(ds)


def load_global_eagle() -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="global",
        vars_of_interest=["u10", "v10", "sp"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


def load_hrrr() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=T0,
        vars_of_interest=["u10", "v10", "sp"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


def load_gfs() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=T0,
        vars_of_interest=["u10", "v10", "sp"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


def load_hrrr_orog() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0="2025-01-07T12",
        vars_of_interest=["orog"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return ds.squeeze()


def load_gfs_orog() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=T0,
        vars_of_interest=["orog"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return ds.squeeze()


# ── Plotting ──────────────────────────────────────────────────────────────────

def _latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon


def plot_panel(ax, ds: xr.Dataset, title: str, map_extent: list, ds_orog: xr.Dataset):
    lat, lon = _latlon(ds)
    wspd = _wspd(ds).values
    # SP expected in Pa; convert to hPa for contour levels
    sp_hpa = ds["sp"].values / 100.0

    im = ax.pcolormesh(
        lon, lat, wspd,
        transform=ccrs.PlateCarree(),
        cmap=cmo.speed_r,
        vmin=0, vmax=WSPD_VMAX,
        shading="auto",
    )
    #cs = ax.contour(
    #    lon, lat, sp_hpa,
    #    levels=SP_LEVELS,
    #    colors="white",
    #    linewidths=0.7,
    #    transform=ccrs.PlateCarree(),
    #)
#    ax.clabel(cs, levels=SP_LEVELS[::2], fmt="%d", fontsize=6, inline=True)

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.4, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="gray")

    lat_o, lon_o = _latlon(ds_orog)
    ax.contour(
        lon_o, lat_o, ds_orog["orog"].values,
        levels=OROG_LEVELS,
        colors="black",
        linewidths=0.4,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(title, loc="left", fontsize=10)

    return im


def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print(f"T0={T0}  FHR={FHR}  valid={VALID_TIME}")

    print("Loading Nested-EAGLE (LAM)...")
    ds_nested = load_nested_eagle()
    print("Loading Global-EAGLE...")
    ds_global = load_global_eagle()
    print("Loading HRRR...")
    try:
        ds_hrrr = load_hrrr()
    except:
        ds_hrrr = None
        print(" ... can't load HRRR for this t0/fhr combo")

    print("Loading HRRR orography...")
    ds_hrrr_orog = load_hrrr_orog()
    print("Loading GFS...")
    ds_gfs = load_gfs()
    print("Loading GFS orography...")
    ds_gfs_orog = load_gfs_orog()

    map_extent = [_WEST, _EAST, _SOUTH, _NORTH]

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    # Panel order matches the time-series figure:
    # Nested-EAGLE | GFS  /  HRRR | Global-EAGLE
    panels = [
        (axes[0, 0], ds_nested, "Nested-EAGLE", ds_hrrr_orog),
        (axes[0, 1], ds_gfs,    "GFS",          ds_gfs_orog),
        (axes[1, 0], ds_hrrr,   "HRRR",         ds_hrrr_orog),
        (axes[1, 1], ds_global, "Global-EAGLE", ds_gfs_orog),
    ]

    for i, (ax, ds, title, ds_orog) in enumerate(panels):
        first = (i == 0)
        if ds is not None:
            im = plot_panel(ax, ds, title, map_extent, ds_orog)
        else:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="gray")
            ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="gray")
            ax.set_title(f"{title} (no data)", loc="left", fontsize=10)
        # Tier-colored station markers on every panel; labels only on the
        # first to avoid clutter (positions are identical across panels).
        draw_stations(ax, label=first, fires=True, fire_label=first, fontsize=6.5)

    axes[0, 0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.04, aspect=50, extend="max", shrink=0.8)
    cbar.set_label("10 m Wind Speed (m s$^{-1}$)")

    fig.suptitle(
        f"10 m wind speed"
        f" — init {T0}, +{FHR}h ({VALID_TIME.strftime('%Y-%m-%d %HZ')})",
        fontsize=12,
    )

    t0_str  = T0.replace("T", "_")
    outfile = f"figures/map_10m_wind_{t0_str}_fhr{FHR:03d}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
