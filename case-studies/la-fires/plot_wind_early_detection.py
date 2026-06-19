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

# Tiers (and which stations are dropped) are defined in query_obs.STATIONS by
# the data-driven peak-3h-mean criterion. Plot Tier A first, then Tier B.
TARGET_STATIONS = (
    [s for s in STATIONS if STATIONS[s]["tier"] == "A"]
    + [s for s in STATIONS if STATIONS[s]["tier"] == "B"]
)
MODELS = ["Nested-EAGLE", "Global-EAGLE", "HRRR", "GFS"]

# 2×2 panel placement
PANEL_LAYOUT = [
    ["Nested-EAGLE", "GFS"],
    ["HRRR",         "Global-EAGLE"],
]

# High-resolution models get a small representativeness envelope (min–max wind
# over grid cells within RADIUS_KM of the station). Radius is tied to obs/jet-
# placement representativeness, NOT to give the coarse models more cells:
# station spacing is 7–17 km, so a larger radius would blur the spatial
# discrimination this analysis is about. Coarse models (GFS, Global-EAGLE) have
# ≤1 cell at this scale by construction, so no band is drawn for them.
HIRES_MODELS = {"Nested-EAGLE", "HRRR"}
RADIUS_KM = 5.0

# Observations: METAR (hourly) + SPECI (irregular, event-triggered) are sampled
# unevenly, so resample onto a regular grid first, then take a 3 h centered
# running mean (±1.5 h) to compare against the coarse-cadence model output.
OBS_RESAMPLE = "30min"
OBS_WINDOW   = "3h"

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


# ── Neighborhood representativeness envelope (high-res models only) ─────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(((lon2 - lon1 + 180) % 360) - 180)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_masks(ds: xr.Dataset, radius_km: float = RADIUS_KM) -> dict:
    """Per-station boolean (y, x) mask of cells within radius_km. Grid is static
    across inits, so this is computed once per model and reused."""
    lat2d = ds["lat"].values
    lon2d = np.where(ds["lon"].values > 180, ds["lon"].values - 360, ds["lon"].values)
    masks = {}
    for s in TARGET_STATIONS:
        d = _haversine_km(STATIONS[s]["lat"], STATIONS[s]["lon"], lat2d, lon2d)
        masks[s] = xr.DataArray(d <= radius_km, dims=("y", "x"))
    return masks


def smooth_obs(stn_df: pd.DataFrame) -> pd.Series:
    """3 h centered running mean of WSPD on a regular OBS_RESAMPLE grid."""
    s = (
        stn_df.set_index("OBS_TIMESTAMP")["WSPD"]
        .sort_index()
        .resample(OBS_RESAMPLE)
        .mean()
    )
    return s.rolling(OBS_WINDOW, center=True, min_periods=1).mean()


def envelope(ds: xr.Dataset, masks: dict) -> tuple[xr.DataArray, xr.DataArray]:
    """min/max wind over each station's neighborhood -> DataArray(time, station)."""
    lo, hi = {}, {}
    for s, m in masks.items():
        wm = ds["wspd"].where(m)
        lo[s] = wm.min(("y", "x"))
        hi[s] = wm.max(("y", "x"))
    lo_da = xr.concat([lo[s] for s in TARGET_STATIONS], dim="station")
    hi_da = xr.concat([hi[s] for s in TARGET_STATIONS], dim="station")
    lo_da["station"] = TARGET_STATIONS
    hi_da["station"] = TARGET_STATIONS
    return lo_da, hi_da


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

    # interp[model][t0]     = DataArray(time, station)  -- bilinear point value
    # interp_env[model][t0] = (lo, hi) DataArrays(time, station)  -- neighborhood
    interp = {m: {} for m in MODELS}
    interp_env = {m: {} for m in HIRES_MODELS}
    eagle_masks = None
    hrrr_masks  = None

    print("\nLoading Nested-EAGLE, Global-EAGLE, and GFS (7 inits)...")
    for t0 in EAGLE_GFS_INITS:
        print(f"  {t0}")
        ds_eagle = load_eagle(t0)
        if eagle_regridder is None:
            print("  Building EAGLE regridder + neighborhood masks...")
            eagle_regridder = xe.Regridder(ds_eagle, target, "bilinear", locstream_out=True)
            eagle_masks = build_masks(ds_eagle)
        interp["Nested-EAGLE"][t0] = interp_lcc(ds_eagle, eagle_regridder)
        interp_env["Nested-EAGLE"][t0] = envelope(ds_eagle, eagle_masks)

        ds_global = load_global_eagle(t0)
        interp["Global-EAGLE"][t0] = interp_gfs(ds_global)

        ds_gfs = load_gfs(t0)
        interp["GFS"][t0] = interp_gfs(ds_gfs)

    print("\nLoading HRRR (3 inits)...")
    for t0 in HRRR_INITS:
        print(f"  {t0}")
        ds_hrrr = load_hrrr(t0)
        if hrrr_regridder is None:
            print("  Building HRRR regridder + neighborhood masks...")
            hrrr_regridder = xe.Regridder(ds_hrrr, target, "bilinear", locstream_out=True)
            hrrr_masks = build_masks(ds_hrrr)
        interp["HRRR"][t0] = interp_lcc(ds_hrrr, hrrr_regridder)
        interp_env["HRRR"][t0] = envelope(ds_hrrr, hrrr_masks)

    # ── One figure per station ────────────────────────────────────────────────
    for icao in TARGET_STATIONS:
        meta = STATIONS[icao]
        stn_df = df[df["RPID"] == icao]

        fig, axes = plt.subplots(
            2, 2,
            figsize=(13, 8),
            sharex=True, sharey=True,
            constrained_layout=True,
        )

        obs_smooth = smooth_obs(stn_df) if len(stn_df) > 0 else None

        for r in range(2):
            for c in range(2):
                ax = axes[r, c]
                model = PANEL_LAYOUT[r][c]
                if len(stn_df) > 0:
                    # raw obs faint in the back, 3 h centered mean as solid line
                    ax.plot(
                        stn_df["OBS_TIMESTAMP"], stn_df["WSPD"],
                        color="gray", lw=0.8, alpha=0.25, zorder=1,
                    )
                    ax.plot(
                        obs_smooth.index, obs_smooth.values,
                        color="gray", lw=2.2, zorder=2,
                    )

                for t0 in MODEL_INITS[model]:
                    da = interp[model][t0].sel(station=icao)
                    times = pd.DatetimeIndex(da.time.values).tz_localize("UTC")
                    if model in HIRES_MODELS:
                        lo, hi = interp_env[model][t0]
                        ax.fill_between(
                            times,
                            lo.sel(station=icao).values,
                            hi.sel(station=icao).values,
                            color=INIT_COLORS[t0], alpha=0.12, lw=0, zorder=3,
                        )
                    ax.plot(times, da.values, color=INIT_COLORS[t0], lw=1.4, zorder=4)

                for val in THRESHOLDS_MS.values():
                    ax.axhline(val, color="gray", lw=0.5, ls=":", alpha=0.6)

                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.3)
                if model in HIRES_MODELS:
                    title = f"{model}  (line: bilinear · band: min–max within {RADIUS_KM:.0f} km)"
                else:
                    title = f"{model}  (single grid cell at station scale)"
                ax.set_title(title, loc="left", fontsize=10)
                if c == 0:
                    ax.set_ylabel("m/s")

        for c in range(2):
            axes[1, c].xaxis.set_major_locator(mdates.DayLocator())
            axes[1, c].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            axes[1, c].set_xlabel("Date (UTC), January 2025")
        axes[1, 0].set_xlim(
            pd.Timestamp("2025-01-06", tz="UTC"),
            pd.Timestamp("2025-01-11", tz="UTC"),
        )

        # Legend: obs + one handle per init date
        obs_handle = Line2D([0], [0], color="gray", lw=2.2, label="obs (3 h mean)")
        init_handles = [
            Line2D([0], [0], color=INIT_COLORS[t0], lw=1.4,
                   label=t0.replace("T12", ""))
            for t0 in EAGLE_GFS_INITS
        ]
        fig.legend(
            handles=[obs_handle] + init_handles,
            loc="outside right upper",
            ncol=1,
            fontsize=8,
            title="init date (12Z)",
            title_fontsize=8,
        )

        fig.suptitle(
            f"{icao} — {meta['name']} (Tier {meta['tier']}) | "
            f"10-m wind speed: obs vs forecasts",
            fontsize=12,
        )

        outfile = f"figures/tier{meta['tier']}_{icao}_wind_early_detection.png"
        fig.savefig(outfile, dpi=150)
        print(f"\nSaved {outfile}")
        plt.close(fig)


if __name__ == "__main__":
    main()
