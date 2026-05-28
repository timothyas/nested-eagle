"""
Overlay forecast 10-m wind speed on observed sustained winds at the SoCal
ASOS stations during the January 2025 Santa Ana / LA fires event.

LCC-grid models (Nested-EAGLE, HRRR) are bilinearly interpolated to station
locations using xesmf (sharing one regridder since they share the same grid).
GFS uses xarray.interp directly on its regular 1-D lat/lon grid.

Color encodes model; linestyle encodes initialization time.
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
from query_obs import STATIONS, THRESHOLDS_MS


def get_color(label):
    if "Nested" in label:
        return "C0"
    elif "HRRR" in label:
        return "C1"
    elif "GFS" in label:
        return "C2"
    elif "Global-EAGLE" in label:
        return "C5"
    return None


OBS_FILE = "socal_wind_obs_jan2025.parquet"
OUTFILE = "figures/socal_wind_fcst_vs_obs.png"

EAGLE_DIR = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/production/gfs-hrrr"
    "/stage1c/inference-testing"
)
HRRR_ZARR = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines"
    "/hrrr-forecasts-vs-hrrr-analysis/hrrr.forecasts.zarr"
)
GFS_ZARR = (
    "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines"
    "/gfs-forecasts-vs-gfs-analysis/gfs.forecasts.zarr"
)

MODELS = ["Nested-EAGLE", "HRRR", "GFS"]

# (init time, linestyle) — add more rows to overlay additional initializations
INITS = [
    ("2025-01-06T06", "-"),
    ("2025-01-03T18", "--"),
    ("2025-01-01T06", ":"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


def load_eagle(t0: str) -> xr.Dataset:
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
    return ds.assign(wspd=_wspd(ds))[["wspd"]].rename(
        {"latitude": "lat", "longitude": "lon"}
    )


def load_hrrr(t0: str) -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=t0,
        vars_of_interest=["u10", "v10"],
        trim_edge=[25, 24, 25, 26],
        load=True,
        reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480},
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]].rename(
        {"latitude": "lat", "longitude": "lon"}
    )


def load_gfs(t0: str) -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR,
        t0=t0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]]


# ── Interpolation ─────────────────────────────────────────────────────────────

def _station_target() -> xr.Dataset:
    return xr.Dataset(
        {
            "lat": xr.DataArray(
                [m["lat"] for m in STATIONS.values()], dims=["station"]
            ),
            "lon": xr.DataArray(
                [m["lon"] % 360 for m in STATIONS.values()], dims=["station"]
            ),
        }
    )


def interp_lcc(ds: xr.Dataset, regridder: xe.Regridder) -> xr.DataArray:
    da = regridder(ds["wspd"])
    da["station"] = list(STATIONS.keys())
    return da


def interp_gfs(ds: xr.Dataset) -> xr.DataArray:
    lats = xr.DataArray([m["lat"] for m in STATIONS.values()], dims=["station"])
    lons = xr.DataArray([m["lon"] % 360 for m in STATIONS.values()], dims=["station"])
    da = ds["wspd"].interp(latitude=lats, longitude=lons, method="linear")
    da["station"] = list(STATIONS.keys())
    return da


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print("Loading observations...")
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    df = df.sort_values("OBS_TIMESTAMP")

    # Load and interpolate all inits; build regridder once from the first eagle
    # dataset since all LCC datasets share the same grid.
    regridder = None
    interp = {}  # interp[t0][model] = DataArray(time, station)

    for t0, _ in INITS:
        print(f"\nLoading init {t0}...")
        ds_eagle = load_eagle(t0)
        ds_hrrr  = load_hrrr(t0)
        ds_gfs   = load_gfs(t0)

        if regridder is None:
            print("Building LCC regridder...")
            regridder = xe.Regridder(
                ds_eagle, _station_target(), "bilinear", locstream_out=True
            )

        interp[t0] = {
            "Nested-EAGLE": interp_lcc(ds_eagle, regridder),
            "HRRR":         interp_lcc(ds_hrrr,  regridder),
            "GFS":          interp_gfs(ds_gfs),
        }

    stations = list(STATIONS.keys())
    n = len(stations)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(11, 1.7 * n),
        sharex=True,
        constrained_layout=True,
    )

    for ax, icao in zip(axes, stations):
        meta = STATIONS[icao]
        stn = df[df["RPID"] == icao]
        stn_idx = stations.index(icao)

        if len(stn) > 0:
            ax.plot(stn["OBS_TIMESTAMP"], stn["WSPD"], color="gray")

        for t0, ls in INITS:
            for model in MODELS:
                da = interp[t0][model].isel(station=stn_idx)
                times = pd.DatetimeIndex(da.time.values).tz_localize("UTC")
                ax.plot(times, da.values, color=get_color(model), ls=ls)

        for val in THRESHOLDS_MS.values():
            ax.axhline(val, color="gray", lw=0.5, ls=":", alpha=0.6)

        ax.set_ylabel("m/s")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{icao} — {meta['name']} (Tier {meta['tier']})",
            loc="left", fontsize=10,
        )

    # Two-part proxy legend: colors for models, linestyles for inits
    color_handles = [Line2D([0], [0], color="gray", label="obs")] + [
        Line2D([0], [0], color=get_color(m), label=m) for m in MODELS
    ]
    ls_handles = [
        Line2D([0], [0], color="black", ls=ls, label=t0) for t0, ls in INITS
    ]
    axes[0].legend(
        handles=color_handles + ls_handles,
        loc="upper right",
        ncol=len(color_handles),
    )

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("Date (UTC), January 2025")

    fig.suptitle("SoCal 10-m wind speed — obs vs forecasts", fontsize=12)

    fig.savefig(OUTFILE, dpi=150)
    print(f"\nSaved {OUTFILE}")


if __name__ == "__main__":
    main()
