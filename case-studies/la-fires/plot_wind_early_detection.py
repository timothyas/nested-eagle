"""
Per-station forecast vs obs wind comparison for KVNY and KBUR.
One figure per station; rows = models (Nested-EAGLE, HRRR, GFS).
Line color encodes initialization time via the inferno colormap so the
progression from early to late forecasts is immediately visible.

GFS and Nested-EAGLE: Jan 1–7 12Z (7 inits, every 24 h).
HRRR: Jan 5–7 12Z (3 inits — limited by 48-h forecast window).
Colors are assigned from the same 7-step inferno scale for all models so
that HRRR's Jan 5–7 colors match those panels in GFS/EAGLE rows.
"""

import os

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from matplotlib.lines import Line2D

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from query_obs import STATIONS, THRESHOLDS_MS

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

OBS_FILE = "socal_bbox_obs_jan2025.parquet"

EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")

TARGET_STATIONS = ["KVNY", "KBUR", "KMWS"]
MODELS = ["Nested-EAGLE", "Global-EAGLE", "HRRR", "GFS"]

# Jan 1–7 12Z every 24 h
EAGLE_GFS_INITS = [f"2025-01-0{d}T12" for d in range(1, 8)]
# Jan 5–7 12Z (48-h HRRR window)
HRRR_INITS = EAGLE_GFS_INITS[-3:]

# Shared inferno color scale: 7 steps, avoiding the very dark/light ends
_inferno = cm.get_cmap("inferno")
N = len(EAGLE_GFS_INITS)
INIT_COLORS = {t0: _inferno(0.15 + 0.70 * i / (N - 1)) for i, t0 in enumerate(EAGLE_GFS_INITS)}

MODEL_INITS = {
    "Nested-EAGLE": EAGLE_GFS_INITS,
    "Global-EAGLE": EAGLE_GFS_INITS,
    "HRRR":         HRRR_INITS,
    "GFS":          EAGLE_GFS_INITS,
}


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


def load_global_eagle(t0: str) -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{t0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path,
        model_type="global",
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]]


def load_hrrr(t0: str) -> xr.Dataset:
    # No trim_edge: full 529×899 HRRR grid; separate regridder built below.
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR,
        t0=t0,
        vars_of_interest=["u10", "v10"],
        load=True,
        reshape_cell_to_2d=True,
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
    return xr.Dataset({
        "lat": xr.DataArray([STATIONS[s]["lat"] for s in TARGET_STATIONS], dims=["station"]),
        "lon": xr.DataArray([STATIONS[s]["lon"] % 360 for s in TARGET_STATIONS], dims=["station"]),
    })


def interp_lcc(ds: xr.Dataset, regridder: xe.Regridder) -> xr.DataArray:
    da = regridder(ds["wspd"])
    da["station"] = TARGET_STATIONS
    return da


def interp_gfs(ds: xr.Dataset) -> xr.DataArray:
    lats = xr.DataArray([STATIONS[s]["lat"] for s in TARGET_STATIONS], dims=["station"])
    lons = xr.DataArray([STATIONS[s]["lon"] % 360 for s in TARGET_STATIONS], dims=["station"])
    da = ds["wspd"].interp(latitude=lats, longitude=lons, method="linear")
    da["station"] = TARGET_STATIONS
    return da


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print("Loading observations...")
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    df = df.sort_values("OBS_TIMESTAMP")

    target = _station_target()
    eagle_regridder = None
    hrrr_regridder  = None

    # interp[model][t0] = DataArray(time, station)
    interp = {m: {} for m in MODELS}

    print("\nLoading Nested-EAGLE, Global-EAGLE, and GFS (7 inits)...")
    for t0 in EAGLE_GFS_INITS:
        print(f"  {t0}")
        ds_eagle = load_eagle(t0)
        if eagle_regridder is None:
            print("  Building EAGLE regridder...")
            eagle_regridder = xe.Regridder(ds_eagle, target, "bilinear", locstream_out=True)
        interp["Nested-EAGLE"][t0] = interp_lcc(ds_eagle, eagle_regridder)

        ds_global = load_global_eagle(t0)
        interp["Global-EAGLE"][t0] = interp_gfs(ds_global)

        ds_gfs = load_gfs(t0)
        interp["GFS"][t0] = interp_gfs(ds_gfs)

    print("\nLoading HRRR (3 inits)...")
    for t0 in HRRR_INITS:
        print(f"  {t0}")
        ds_hrrr = load_hrrr(t0)
        if hrrr_regridder is None:
            print("  Building HRRR regridder...")
            hrrr_regridder = xe.Regridder(ds_hrrr, target, "bilinear", locstream_out=True)
        interp["HRRR"][t0] = interp_lcc(ds_hrrr, hrrr_regridder)

    # ── One figure per station ────────────────────────────────────────────────
    for icao in TARGET_STATIONS:
        meta = STATIONS[icao]
        stn_df = df[df["RPID"] == icao]

        fig, axes = plt.subplots(
            len(MODELS), 1,
            figsize=(11, 3.5 * len(MODELS)),
            sharex=True,
            constrained_layout=True,
        )

        for ax, model in zip(axes, MODELS):
            if len(stn_df) > 0:
                ax.plot(
                    stn_df["OBS_TIMESTAMP"], stn_df["WSPD"],
                    color="gray", lw=2.0, zorder=10,
                )

            for t0 in MODEL_INITS[model]:
                da = interp[model][t0].sel(station=icao)
                times = pd.DatetimeIndex(da.time.values).tz_localize("UTC")
                ax.plot(times, da.values, color=INIT_COLORS[t0], lw=1.4)

            for val in THRESHOLDS_MS.values():
                ax.axhline(val, color="gray", lw=0.5, ls=":", alpha=0.6)

            ax.set_ylabel("m/s")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.set_title(model, loc="left", fontsize=10)

        axes[-1].xaxis.set_major_locator(mdates.DayLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axes[-1].set_xlabel("Date (UTC), January 2025")

        # Legend: obs + one handle per init date
        obs_handle = Line2D([0], [0], color="gray", lw=2.0, label="obs")
        init_handles = [
            Line2D([0], [0], color=INIT_COLORS[t0], lw=1.4,
                   label=t0.replace("T12", ""))
            for t0 in EAGLE_GFS_INITS
        ]
        axes[0].legend(
            handles=[obs_handle] + init_handles,
            loc="upper right",
            ncol=len(init_handles) + 1,
            fontsize=8,
            title="init date (12Z)",
            title_fontsize=8,
        )

        fig.suptitle(
            f"{icao} — {meta['name']} | 10-m wind speed: obs vs forecasts",
            fontsize=12,
        )

        outfile = f"figures/{icao}_wind_early_detection.png"
        fig.savefig(outfile, dpi=150)
        print(f"\nSaved {outfile}")
        plt.close(fig)


if __name__ == "__main__":
    main()
