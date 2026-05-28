"""
2×2 map of 10 m wind speed (cmo.speed colormap) and surface pressure
(4 hPa contours) comparing Nested-EAGLE (LAM, native grid), Global-EAGLE,
HRRR, and GFS.
LambertConformal projection centred on Southern California.
Western extent matches the HRRR data boundary at SoCal latitudes.

Edit T0 and FHR to change the plotted forecast.
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

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

# ── Parameters ────────────────────────────────────────────────────────────────
T0  = "2025-01-06T12"
FHR = 24

VALID_TIME = pd.Timestamp(T0) + pd.Timedelta(hours=FHR)
SP_LEVELS  = np.arange(972, 1024, 4)   # surface pressure contour levels (hPa)
WSPD_VMAX  = 10.0                       # m/s, shared colorscale

# SoCal fixed bounds (east/north/south); west is computed from HRRR at load time
_EAST  = -113.5
_NORTH =   40
_SOUTH =   30

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


def _hrrr_west_extent(ds: xr.Dataset) -> float:
    """Minimum longitude of the HRRR grid within the SoCal latitude band."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    lon = np.where(lon > 180, lon - 360, lon)
    mask = (lat >= _SOUTH) & (lat <= _NORTH)
    return float(lon[mask].min())


# ── Plotting ──────────────────────────────────────────────────────────────────

def _latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    lon = np.where(lon > 180, lon - 360, lon)
    return lat, lon


def plot_panel(ax, ds: xr.Dataset, title: str, map_extent: list):
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
    ds_hrrr = load_hrrr()
    print("Loading GFS...")
    ds_gfs = load_gfs()

    west = _hrrr_west_extent(ds_hrrr)
    map_extent = [west, _EAST, _SOUTH, _NORTH]
    print(f"Map extent: {map_extent}")

    proj = ccrs.LambertConformal(central_longitude=-118, central_latitude=34,
                                 standard_parallels=(33, 45))
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    panels = [
        (axes[0, 0], ds_nested, "Nested-EAGLE"),
        (axes[0, 1], ds_global, "Global-EAGLE"),
        (axes[1, 0], ds_hrrr,   "HRRR"),
        (axes[1, 1], ds_gfs,    "GFS"),
    ]

    for ax, ds, title in panels:
        im = plot_panel(ax, ds, title, map_extent)

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
