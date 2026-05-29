"""
2×2 spatial map of 10-m wind speed forecast error at SoCal ASOS locations.

Signed error = forecast − obs (3-h rolling mean applied to obs;
nearest obs to the valid time is selected per station).
Dot color: diverging RdBu_r (blue = under-forecast, red = over-forecast).
Dot size: proportional to |error|.

LCC-grid models (Nested-EAGLE, HRRR) use xesmf bilinear interpolation;
both share one regridder since they live on the same grid.
Regular lat/lon models (Global-EAGLE, GFS) use xarray.interp.

Edit T0 and FHR to change the plotted forecast.
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset

SCRATCH  = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

# ── Parameters ────────────────────────────────────────────────────────────────
T0  = "2025-01-06T12"
FHR = 24

VALID_TIME    = pd.Timestamp(T0) + pd.Timedelta(hours=FHR)
ROLLING_HOURS = 3
ERR_VMAX      = 5.0   # m/s, symmetric colorbar limits
DOT_SCALE     = 30    # marker area per m/s of |error|
DOT_MIN       = 5     # minimum marker area (keeps zero-error dots visible)

# Map bounds
_WEST  = -120.5
_EAST  = -116.0
_NORTH =   35.8
_SOUTH =   32.5

# ── Data paths ────────────────────────────────────────────────────────────────
OBS_FILE         = "socal_bbox_obs_jan2025.parquet"
EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")


# ── Observation loading ───────────────────────────────────────────────────────

def load_obs() -> pd.DataFrame:
    """Rolling-mean obs, one value per station nearest to VALID_TIME."""
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)

    records = []
    for rpid, grp in df.groupby("RPID"):
        grp = grp.sort_values("OBS_TIMESTAMP").set_index("OBS_TIMESTAMP")
        wspd_smooth = grp["WSPD"].rolling(
            f"{ROLLING_HOURS}h", center=True, min_periods=1
        ).mean()
        valid_utc = VALID_TIME.tz_localize("UTC")
        idx = np.argmin(np.abs((wspd_smooth.index - valid_utc).total_seconds().values))
        records.append({
            "RPID": rpid,
            "LAT":  float(grp["LAT"].iloc[0]),
            "LON":  float(grp["LON"].iloc[0]),
            "WSPD": float(wspd_smooth.iloc[idx]),
        })

    return pd.DataFrame(records).dropna(subset=["WSPD"])


# ── Forecast loading ──────────────────────────────────────────────────────────

def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


def _select(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(time=VALID_TIME).squeeze()


def load_nested_eagle() -> xr.Dataset:
    path = os.path.join(EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="nested-lam",
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480},
        lam_index=407040,
    )
    return _select(
        ds.assign(wspd=_wspd(ds))[["wspd"]].rename({"latitude": "lat", "longitude": "lon"})
    )


def load_global_eagle() -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="global",
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds.assign(wspd=_wspd(ds))[["wspd"]])


def load_hrrr() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=T0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
        trim_edge=[25, 24, 25, 26],
    )
    return _select(
        ds.assign(wspd=_wspd(ds))[["wspd"]].rename({"latitude": "lat", "longitude": "lon"})
    )


def load_gfs() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=T0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _select(ds.assign(wspd=_wspd(ds))[["wspd"]])


# ── Interpolation ─────────────────────────────────────────────────────────────

def _lcc_target(df_obs: pd.DataFrame) -> xr.Dataset:
    return xr.Dataset({
        "lat": xr.DataArray(df_obs["LAT"].values, dims=["station"]),
        "lon": xr.DataArray(df_obs["LON"].values % 360, dims=["station"]),
    })


def interp_lcc(ds: xr.Dataset, regridder: xe.Regridder) -> np.ndarray:
    return regridder(ds["wspd"]).values


def interp_latlon(ds: xr.Dataset, df_obs: pd.DataFrame) -> np.ndarray:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = xr.DataArray(df_obs["LAT"].values, dims=["station"])
    lons = xr.DataArray(df_obs["LON"].values % 360, dims=["station"])
    return ds["wspd"].interp(
        {lat_name: lats, lon_name: lons}, method="linear"
    ).values


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, lons, lats, errors, title):
    sc = ax.scatter(
        lons, lats,
        c=errors,
        s=np.abs(errors) * DOT_SCALE + DOT_MIN,
        cmap="RdBu_r", vmin=-ERR_VMAX, vmax=ERR_VMAX,
        transform=ccrs.PlateCarree(),
        edgecolors="black", linewidths=0.3, zorder=5,
    )
    ax.set_extent([_WEST, _EAST, _SOUTH, _NORTH], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS,   lw=0.4, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.STATES,    lw=0.3, edgecolor="gray")
    ax.set_title(title, loc="left", fontsize=10)
    return sc


def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print(f"T0={T0}  FHR={FHR}  valid={VALID_TIME}")

    print("Loading observations...")
    df_obs = load_obs()
    print(f"  {len(df_obs)} stations with valid obs")

    print("Loading Nested-EAGLE...")
    ds_nested = load_nested_eagle()
    print("Loading Global-EAGLE...")
    ds_global = load_global_eagle()
    print("Loading HRRR...")
    ds_hrrr   = load_hrrr()
    print("Loading GFS...")
    ds_gfs    = load_gfs()

    target = _lcc_target(df_obs)
    print("Building LCC regridder (Nested-EAGLE / HRRR)...")
    lcc_regridder = xe.Regridder(ds_nested, target, "bilinear", locstream_out=True)

    obs  = df_obs["WSPD"].values
    lats = df_obs["LAT"].values
    lons = df_obs["LON"].values

    panels = [
        ("Nested-EAGLE", interp_lcc(ds_nested, lcc_regridder) - obs),
        ("Global-EAGLE", interp_latlon(ds_global, df_obs)     - obs),
        ("HRRR",         interp_lcc(ds_hrrr,   lcc_regridder) - obs),
        ("GFS",          interp_latlon(ds_gfs,  df_obs)        - obs),
    ]

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    sc = None
    for ax, (title, errors) in zip(axes.flatten(), panels):
        sc = plot_panel(ax, lons, lats, errors, title)

    cbar = fig.colorbar(
        sc, ax=axes, orientation="horizontal",
        fraction=0.05, pad=0.04, aspect=50, shrink=0.8, extend="both",
    )
    cbar.set_label("10-m wind speed error: forecast − obs (m s$^{-1}$)")

    # Dot size reference in bottom-right panel
    ax_ref = axes[1, 1]
    for ref in [1, 3, 5]:
        ax_ref.scatter(
            [], [], c="gray",
            s=ref * DOT_SCALE + DOT_MIN,
            edgecolors="black", linewidths=0.3,
            label=f"{ref} m s$^{{-1}}$",
            transform=ccrs.PlateCarree(),
        )
    ax_ref.legend(title="|error|", loc="lower right", fontsize=7, title_fontsize=7)

    fig.suptitle(
        f"10-m wind speed forecast error"
        f" — init {T0}, +{FHR}h ({VALID_TIME.strftime('%Y-%m-%d %HZ')})",
        fontsize=12,
    )

    t0_str  = T0.replace("T", "_")
    outfile = f"figures/map_fcst_error_{t0_str}_fhr{FHR:03d}.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
