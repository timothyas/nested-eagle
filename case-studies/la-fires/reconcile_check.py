"""
Reconciliation diagnostic for the map-vs-timeseries inconsistency.

For the exact case shown in plot_map_10m_wind.py (T0=2025-01-06T12, FHR=48),
report for every station (Tier A + Tier B) and all four models:

  bilinear : point value via the SAME interpolation the early-detection
             timeseries uses (xesmf bilinear for curvilinear grids,
             xr.interp linear for regular grids)
  nearest  : nearest-grid-cell value (what the map pcolormesh shows)
  boxmax   : max wind within ~RADIUS_KM of the station
  boxmean  : mean wind within ~RADIUS_KM of the station
  obs      : observed sustained wind nearest the valid time

Step 1 (rule out a bug): bilinear should be close to nearest. A large
mismatch means a coordinate/interp problem, not physics.

Step 2 (everywhere vs skill): compare GFS across Tier A and Tier B. If GFS
runs hot at the coastal Tier-B controls too, its Tier-A "hits" are spurious.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from eagle.tools.data import open_anemoi_inference_dataset, open_forecast_zarr_dataset
from query_obs import STATIONS

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")

T0 = "2025-01-06T12"
FHR = 48
VALID_TIME = pd.Timestamp(T0) + pd.Timedelta(hours=FHR)
RADIUS_KM = 12.0

OBS_FILE = "socal_bbox_obs_jan2025.parquet"

EAGLE_DIR        = os.path.join(LA_FIRES, "nested-eagle")
GLOBAL_EAGLE_DIR = os.path.join(LA_FIRES, "global-eagle")
HRRR_ZARR        = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
GFS_ZARR         = os.path.join(LA_FIRES, "gfs.forecasts.zarr")

# Order: Tier A first, then Tier B
ALL_STATIONS = (
    [s for s in STATIONS if STATIONS[s]["tier"] == "A"]
    + [s for s in STATIONS if STATIONS[s]["tier"] == "B"]
)


def _wspd(ds: xr.Dataset) -> xr.DataArray:
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


# ── Loaders (mirror the two plotting scripts) ──────────────────────────────────

def load_nested_eagle() -> xr.Dataset:
    path = os.path.join(EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path, model_type="nested-lam",
        vars_of_interest=["u10", "v10"], load=True, reshape_cell_to_2d=True,
        lcc_info={"n_x": 848, "n_y": 480}, lam_index=407040,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]].sel(time=VALID_TIME).squeeze()


def load_global_eagle() -> xr.Dataset:
    path = os.path.join(GLOBAL_EAGLE_DIR, f"{T0}.360h.nc")
    ds = open_anemoi_inference_dataset(
        path, model_type="global",
        vars_of_interest=["u10", "v10"], load=True, reshape_cell_to_2d=True,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]].sel(time=VALID_TIME).squeeze()


def load_hrrr() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR, t0=T0,
        vars_of_interest=["u10", "v10"], load=True, reshape_cell_to_2d=True,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]].sel(time=VALID_TIME).squeeze()


def load_gfs() -> xr.Dataset:
    ds = open_forecast_zarr_dataset(
        path=GFS_ZARR, t0=T0,
        vars_of_interest=["u10", "v10"], load=True, reshape_cell_to_2d=True,
    )
    return ds.assign(wspd=_wspd(ds))[["wspd"]].sel(time=VALID_TIME).squeeze()


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _coord_names(ds: xr.Dataset):
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    return lat, lon


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(((lon2 - lon1 + 180) % 360) - 180)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def sample_regular(ds, slat, slon):
    """GFS / Global-EAGLE: 1D regular lat/lon grid."""
    latn, lonn = _coord_names(ds)
    lon_q = slon % 360
    bil = float(ds["wspd"].interp({latn: slat, lonn: lon_q}, method="linear").values)

    lat1d = ds[latn].values
    lon1d = ds[lonn].values
    lon1d_pm = np.where(lon1d > 180, lon1d - 360, lon1d)
    i = int(np.abs(lat1d - slat).argmin())
    j = int(np.abs(lon1d_pm - slon).argmin())
    near = float(ds["wspd"].isel({latn: i, lonn: j}).values)

    LON, LAT = np.meshgrid(lon1d_pm, lat1d)
    d = _haversine_km(slat, slon, LAT, LON)
    mask = d <= RADIUS_KM
    w = ds["wspd"].values
    return bil, near, float(np.nanmax(w[mask])), float(np.nanmean(w[mask])), int(mask.sum())


def sample_curvilinear(ds, regridder, slat, slon):
    """Nested-EAGLE / HRRR: 2D curvilinear lat/lon grid."""
    latn, lonn = _coord_names(ds)
    bil = float(regridder(ds["wspd"]).values.ravel()[0])  # single-point locstream

    lat2d = ds[latn].values
    lon2d = ds[lonn].values
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    d = _haversine_km(slat, slon, lat2d, lon2d)
    near = float(ds["wspd"].values.ravel()[int(d.argmin())])
    mask = d <= RADIUS_KM
    w = ds["wspd"].values
    return bil, near, float(np.nanmax(w[mask])), float(np.nanmean(w[mask])), int(mask.sum())


def make_regridder(ds, slat, slon):
    latn, lonn = _coord_names(ds)
    tgt = xr.Dataset({
        "lat": xr.DataArray([slat], dims=["station"]),
        "lon": xr.DataArray([slon % 360], dims=["station"]),
    })
    src = ds.rename({latn: "lat", lonn: "lon"}) if (latn, lonn) != ("lat", "lon") else ds
    return xe.Regridder(src, tgt, "bilinear", locstream_out=True)


# ── Obs ────────────────────────────────────────────────────────────────────────

def obs_at(df, icao):
    sub = df[df["RPID"] == icao]
    if len(sub) == 0:
        return np.nan
    dt = (sub["OBS_TIMESTAMP"] - VALID_TIME.tz_localize("UTC")).abs()
    row = sub.loc[dt.idxmin()]
    if abs((row["OBS_TIMESTAMP"] - VALID_TIME.tz_localize("UTC")).total_seconds()) > 3600:
        return np.nan
    return float(row["WSPD"])


def main():
    print(f"T0={T0}  FHR={FHR}  valid={VALID_TIME}  radius={RADIUS_KM} km\n")

    print("Loading models...")
    ds = {
        "Nested-EAGLE": load_nested_eagle(),
        "Global-EAGLE": load_global_eagle(),
        "HRRR":         load_hrrr(),
        "GFS":          load_gfs(),
    }
    regular = {"Global-EAGLE", "GFS"}

    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)

    for model in ds:
        d = ds[model]
        print(f"\n========== {model} ==========")
        print(f"  grid coords: {list(d.coords)}")
        print(f"{'stn':6} {'tier':4} {'obs':>6} {'bilin':>7} {'near':>7} "
              f"{'boxmax':>7} {'boxmean':>7} {'ncells':>6}  d(bil-near)")
        for icao in ALL_STATIONS:
            slat, slon = STATIONS[icao]["lat"], STATIONS[icao]["lon"]
            tier = STATIONS[icao]["tier"]
            o = obs_at(df, icao)
            if model in regular:
                bil, near, bmax, bmean, n = sample_regular(d, slat, slon)
            else:
                rg = make_regridder(d, slat, slon)
                bil, near, bmax, bmean, n = sample_curvilinear(d, rg, slat, slon)
            print(f"{icao:6} {tier:4} {o:6.1f} {bil:7.2f} {near:7.2f} "
                  f"{bmax:7.2f} {bmean:7.2f} {n:6d}  {bil-near:+.2f}")


if __name__ == "__main__":
    main()
