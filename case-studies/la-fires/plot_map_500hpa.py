"""
2×2 map of 500 hPa wind speed (cmo.tempo colormap) and geopotential height
(60 m contours) comparing Nested-EAGLE, Global-EAGLE, HRRR, and GFS.
Robinson projection, bounded 15–60 N, 150–100 W.

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
from eagle.tools.nested import prepare_regrid_target_mask

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")
MESH_DIR = os.path.join(SCRATCH, "nested-eagle/0.25deg-06km/mesh-gen/csmswt-trim25")

# ── Parameters ────────────────────────────────────────────────────────────────
T0  = "2025-01-02T12"
FHR = 0

VALID_TIME = pd.Timestamp(T0) + pd.Timedelta(hours=FHR)
GH_LEVELS  = np.arange(4800, 6360, 60)   # 500 hPa geopotential height (m)
WSPD_VMAX  = 40.0                          # m/s, shared colorscale
MAP_EXTENT = [-150, -100, 15, 60]

# ── Data paths ────────────────────────────────────────────────────────────────
EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")

# ── Nested-EAGLE regrid config ────────────────────────────────────────────────
_anemoi_ref_kwargs = {
    "cutout": [
        {
            "dataset": "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/hrrr.zarr",
            "trim_edge": [25, 24, 25, 26],
        },
        {
            "dataset": "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/gfs.zarr",
        },
    ],
    "adjust": "all",
    "min_distance_km": 6,
}

_forecast_regrid_kwargs = {
    "target_grid_path": os.path.join(MESH_DIR, "global_quarter_degree_with_mask.nc"),
    "regridder_kwargs": {
        "method": "conservative_normed",
        "reuse_weights": True,
        "filename": os.path.join(MESH_DIR, "conservative_lam_to_global.nc"),
    },
}

_forecast_regrid_kwargs["target_grid_path"], _ = prepare_regrid_target_mask(
    anemoi_reference_dataset_kwargs=_anemoi_ref_kwargs,
    horizontal_regrid_kwargs=_forecast_regrid_kwargs,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)


def _select(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=VALID_TIME).squeeze()


def load_nested_eagle() -> xr.Dataset:
    path = os.path.join(EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="nested-global",
        levels=[500],
        vars_of_interest=["gh", "u", "v"],
        load=True,
        reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480},
        lam_index=407040,
        horizontal_regrid_kwargs=_forecast_regrid_kwargs,
    )
    return _select(ds)


def load_global_eagle() -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="global",
        levels=[500],
        vars_of_interest=["gh", "u", "v"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


def load_hrrr() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=T0,
        levels=[500],
        vars_of_interest=["gh", "u", "v"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


def load_gfs() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=T0,
        levels=[500],
        vars_of_interest=["gh", "u", "v"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _latlon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    return ds[lat_name].values, ds[lon_name].values


def _stream_data(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon_1d, lat_1d, u, v) suitable for ax.streamplot.

    streamplot requires a regular 1-D grid. For LCC datasets (HRRR) the
    lat/lon coords are 2-D; we approximate a regular grid by taking the
    centre column for lat and the centre row for lon — acceptable for
    visualisation on a Lambert Conformal domain.  Latitude is flipped to
    ascending order if needed.
    """
    lat, lon = _latlon(ds)
    u = ds["u"].values
    v = ds["v"].values
    if lat.ndim == 2:
        ny, nx = lat.shape
        lat_1d = lat[:, nx // 2]
        lon_1d = lon[ny // 2, :]
    else:
        lat_1d, lon_1d = lat, lon
    if lat_1d[0] > lat_1d[-1]:
        lat_1d = lat_1d[::-1]
        u = u[::-1]
        v = v[::-1]
    return lon_1d, lat_1d, u, v


def plot_panel(ax, ds: xr.Dataset, title: str):
    lat, lon = _latlon(ds)
    wspd = _wspd(ds).values
    gh   = ds["gh"].values

    im = ax.pcolormesh(
        lon, lat, wspd,
        transform=ccrs.PlateCarree(),
        cmap=cmo.tempo_r,
        vmin=0, vmax=WSPD_VMAX,
        shading="auto",
    )
    cs = ax.contour(
        lon, lat, gh,
        levels=GH_LEVELS,
        colors="white",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
    )
    # Label every third contour to avoid clutter
    ax.clabel(cs, levels=GH_LEVELS[::3], fmt="%d", fontsize=6, inline=True)

    #slon, slat, su, sv = _stream_data(ds)
    #ax.streamplot(
    #    slon, slat, su, sv,
    #    transform=ccrs.PlateCarree(),
    #    color="white",
    #    linewidth=0.6,
    #    density=1.5,
    #    arrowsize=0.7,
    #)

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, lw=0.4, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="gray")
    ax.set_title(title, loc="left", fontsize=10)

    return im


def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print(f"T0={T0}  FHR={FHR}  valid={VALID_TIME}")

    print("Loading Nested-EAGLE...")
    ds_nested = load_nested_eagle()
    print("Loading Global-EAGLE...")
    ds_global = load_global_eagle()
    print("Loading HRRR...")
    try:
        ds_hrrr = load_hrrr()
    except:
        print("couldnt do it")
        ds_hrrr = None
    print("Loading GFS...")
    ds_gfs = load_gfs()

    proj = ccrs.Robinson()
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
        if ds is not None:
            im = plot_panel(ax, ds, title)

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50, extend="max", shrink=.8)
    cbar.set_label("500 hPa Wind Speed (m s$^{-2}$)")

    fig.suptitle(
        f"500 hPa wind speed & geopotential height"
        f" — init {T0}, +{FHR}h ({VALID_TIME.strftime('%Y-%m-%d %HZ')})",
        fontsize=12,
    )

    t0_str  = T0.replace("T", "_")
    outfile = f"figures/map_500hpa_{t0_str}_fhr{FHR:03d}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
