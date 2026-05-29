"""
Time series of spatially-averaged RMSE for SoCal ASOS stations during the
January 2025 Santa Ana / LA fires event.

For each model and initialization time, RMSE is computed at 6-hourly valid
times between Jan 6 12Z and Jan 9 12Z, averaging over all stations in the
SoCal bbox obs dataset. Observations use a 3-h rolling mean.

Lines are colored by model (get_color) with alpha decreasing for older inits.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from matplotlib.lines import Line2D

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset

SCRATCH  = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

# ── Parameters ────────────────────────────────────────────────────────────────
VALID_TIMES   = pd.date_range("2025-01-06T12", "2025-01-09T12", freq="6h")
ROLLING_HOURS = 3

OBS_FILE         = "socal_bbox_obs_jan2025.parquet"
EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")

MODELS          = ["Nested-EAGLE", "Global-EAGLE", "HRRR", "GFS"]
EAGLE_GFS_INITS = [f"2025-01-0{d}T12" for d in range(1, 8)]
HRRR_INITS      = EAGLE_GFS_INITS[-3:]   # Jan 5–7 (48-h window)

MODEL_INITS = {
    "Nested-EAGLE": EAGLE_GFS_INITS,
    "Global-EAGLE": EAGLE_GFS_INITS,
    "HRRR":         HRRR_INITS,
    "GFS":          EAGLE_GFS_INITS,
}

N = len(EAGLE_GFS_INITS)
INIT_ALPHA = {t0: 0.25 + 0.75 * i / (N - 1) for i, t0 in enumerate(EAGLE_GFS_INITS)}


def get_color(label):
    if "Nested" in label:       return "C0"
    elif "HRRR"  in label:      return "C1"
    elif "GFS"   in label:      return "C2"
    elif "Global-EAGLE" in label: return "C5"
    return None


# ── Observation matrix ────────────────────────────────────────────────────────

def build_obs_matrix() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns:
        obs_matrix : (n_valid_times, n_stations) float array of rolling-mean WSPD
        coords_df  : DataFrame with LAT / LON columns, index = RPID
    """
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    valid_utc = VALID_TIMES.tz_localize("UTC")

    station_vals   = {}
    station_coords = {}

    for rpid, grp in df.groupby("RPID"):
        grp = grp.sort_values("OBS_TIMESTAMP").set_index("OBS_TIMESTAMP")
        wspd_smooth = grp["WSPD"].rolling(
            f"{ROLLING_HOURS}h", center=True, min_periods=1
        ).mean()

        vals = []
        for vt in valid_utc:
            idx = np.argmin(
                np.abs((wspd_smooth.index - vt).total_seconds().values)
            )
            vals.append(float(wspd_smooth.iloc[idx]))

        station_vals[rpid]   = vals
        station_coords[rpid] = {"LAT": float(grp["LAT"].iloc[0]),
                                 "LON": float(grp["LON"].iloc[0])}

    obs_df    = pd.DataFrame(station_vals,   index=VALID_TIMES)
    coords_df = pd.DataFrame(station_coords).T
    return obs_df.values, coords_df


# ── Forecast loading ──────────────────────────────────────────────────────────

def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


def _keep_valid_times(ds: xr.Dataset) -> xr.Dataset:
    """Retain only model time steps that fall within VALID_TIMES."""
    fcst_times = pd.DatetimeIndex(ds.time.values)
    keep = fcst_times[fcst_times.isin(VALID_TIMES)]
    return ds.sel(time=keep)


def load_nested_eagle(t0: str) -> xr.Dataset:
    path = os.path.join(EAGLE_DIR, f"{t0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="nested-lam",
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480},
        lam_index=407040,
    )
    return _keep_valid_times(
        ds.assign(wspd=_wspd(ds))[["wspd"]].rename({"latitude": "lat", "longitude": "lon"})
    )


def load_global_eagle(t0: str) -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{t0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="global",
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _keep_valid_times(ds.assign(wspd=_wspd(ds))[["wspd"]])


def load_hrrr(t0: str) -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=t0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
        trim_edge=[25, 24, 25, 26],
    )
    return _keep_valid_times(
        ds.assign(wspd=_wspd(ds))[["wspd"]].rename({"latitude": "lat", "longitude": "lon"})
    )


def load_gfs(t0: str) -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=t0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return _keep_valid_times(ds.assign(wspd=_wspd(ds))[["wspd"]])


# ── Interpolation ─────────────────────────────────────────────────────────────

def _lcc_target(coords_df: pd.DataFrame) -> xr.Dataset:
    return xr.Dataset({
        "lat": xr.DataArray(coords_df["LAT"].values, dims=["station"]),
        "lon": xr.DataArray(coords_df["LON"].values % 360, dims=["station"]),
    })


def interp_lcc(ds: xr.Dataset, regridder: xe.Regridder) -> np.ndarray:
    """Returns (n_times, n_stations)."""
    out = regridder(ds["wspd"])
    return out.values if out.ndim == 2 else out.values[np.newaxis, :]


def interp_latlon(ds: xr.Dataset, coords_df: pd.DataFrame) -> np.ndarray:
    """Returns (n_times, n_stations)."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lats = xr.DataArray(coords_df["LAT"].values, dims=["station"])
    lons = xr.DataArray(coords_df["LON"].values % 360, dims=["station"])
    out = ds["wspd"].interp({lat_name: lats, lon_name: lons}, method="linear")
    return out.values if out.ndim == 2 else out.values[np.newaxis, :]


# ── RMSE ──────────────────────────────────────────────────────────────────────

def rmse_per_timestep(fcst: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """(n_times, n_stations) → (n_times,) RMSE ignoring NaN."""
    return np.sqrt(np.nanmean((fcst - obs) ** 2, axis=1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print("Building obs matrix...")
    obs_matrix, coords_df = build_obs_matrix()
    print(f"  {len(VALID_TIMES)} valid times × {coords_df.shape[0]} stations")

    target = _lcc_target(coords_df)
    print("Building LCC regridder...")
    ds_ref = load_nested_eagle(EAGLE_GFS_INITS[-1])   # use last init — guaranteed in range
    lcc_regridder = xe.Regridder(ds_ref, target, "bilinear", locstream_out=True)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for model in MODELS:
        for t0 in MODEL_INITS[model]:
            print(f"  {model}  {t0}")

            if model == "Nested-EAGLE":
                ds   = load_nested_eagle(t0)
                fcst = interp_lcc(ds, lcc_regridder)
            elif model == "Global-EAGLE":
                ds   = load_global_eagle(t0)
                fcst = interp_latlon(ds, coords_df)
            elif model == "HRRR":
                ds   = load_hrrr(t0)
                fcst = interp_lcc(ds, lcc_regridder)
            else:
                ds   = load_gfs(t0)
                fcst = interp_latlon(ds, coords_df)

            if ds.sizes.get("time", 0) == 0:
                continue

            # Align obs rows to the forecast's valid times
            fcst_times = pd.DatetimeIndex(ds.time.values)
            obs_idx    = VALID_TIMES.get_indexer(fcst_times)
            valid      = obs_idx >= 0
            fcst_times = fcst_times[valid]
            fcst       = fcst[valid, :]
            obs_sub    = obs_matrix[obs_idx[valid], :]

            rmse = rmse_per_timestep(fcst, obs_sub)
            ax.plot(
                fcst_times, rmse,
                color=get_color(model),
                alpha=INIT_ALPHA[t0],
                lw=1.4,
            )

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlim(VALID_TIMES[0], VALID_TIMES[-1])
    ax.set_xlabel("Valid time (UTC), January 2025")
    ax.set_ylabel("RMSE (m s$^{-1}$)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    model_handles = [
        Line2D([0], [0], color=get_color(m), lw=2.0, label=m)
        for m in MODELS
    ]
    ax.legend(handles=model_handles, loc="upper right")

    fig.suptitle(
        f"10-m wind RMSE vs SoCal ASOS ({ROLLING_HOURS}-h rolling mean obs)\n"
        "Jan 6 12Z – Jan 9 12Z  |  inits Jan 1–7 12Z (darker = more recent)",
        fontsize=11,
    )

    outfile = "figures/rmse_timeseries_jan2025.png"
    fig.savefig(outfile, dpi=150)
    print(f"\nSaved {outfile}")


if __name__ == "__main__":
    main()
