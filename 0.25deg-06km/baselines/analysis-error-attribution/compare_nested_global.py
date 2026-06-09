"""
Compare Nested-EAGLE vs Global-EAGLE *24 h forecasts* of 2 m temperature and
10 m wind speed at the same obs station locations, over the test year.

This is the fhr=24 ML-forecast analog of the fhr=0 analysis comparison in
``compare_gfs_hrrr.py``. The two ML models differ in their training data:

  - Nested-EAGLE : trained on HRRR over CONUS + GFS elsewhere ("nested-lam")
  - Global-EAGLE : trained on GFS analysis only             ("global")

Both forecasts are valid at the same time (t0 + 24 h), so their *difference* is
directly informative -- the part of the 24 h forecast that is purely a
Nested<->Global representation difference. Sign convention everywhere:

  d = Nested - Global            (positive => Nested larger/warmer/windier)

mirroring HRRR - GFS in the analysis study (Nested is the HRRR-informed model,
Global is the GFS-only model). The scientific question is whether the
analysis-time HRRR-GFS surface signature (lapse-rate 2 m T, roughness 10 m wind)
*persists into the 24 h ML forecast difference*.

Per station and field, over the test-year cycles, with d = Nested - Global at
fhr=24:

  - bias        : mean_t(d)                 (+ => Nested larger)
  - rmse        : sqrt(mean_t(d^2))
  - var_diff    : var_t(d)                  (MSE = bias^2 + var_diff)
  - var_nested  : var_t(Nested)             (each model's own temporal variability)
  - var_global  : var_t(Global)
  - mean_nested : mean_t(Nested)            (context for the bias)
  - mean_global : mean_t(Global)
  - count       : number of cycles contributing

Wind is the scalar 10 m wind speed sqrt(u^2 + v^2), formed on each model's native
grid before interpolation, matching the analysis pipeline.

Predictors are REUSED, not recomputed: the terrain each model effectively sees is
HRRR (Nested, over CONUS) vs GFS (Global), i.e. exactly the orog_diff/slope_diff
already in ``gfs_vs_hrrr.topo.nc`` and the rough_diff in
``gfs_vs_hrrr.roughness.nc``. The station set and ordering are reproduced from
``compare_gfs_hrrr.py`` (same STATION_FILE + HRRR-domain mask) so this metrics
file lines up 1:1 with those predictor files.

Output (to OUT_DIR, same dir as the analysis study):
  nested_vs_global.fhr24.metrics.nc  -- per-station metrics (dims: field, station)

Heavy step: meant to run on an interactive CPU node (per-file we open lazily,
select only the fhr=24 valid time, then load -- so each 28/20 GB file only ever
materializes one lead x 3 vars).
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr
import pandas as pd

from eagle.tools.data import open_anemoi_inference_dataset

# Reuse the analysis-study machinery so stations, regridding and grid handling
# are identical (and the outputs align 1:1 with gfs_vs_hrrr.topo.nc).
from compare_gfs_hrrr import (
    load_stations, gfs_grid, hrrr_grid, make_regridder, regrid_fields,
    hrrr_valid_mask, station_out, to_station, FIELDS, VALID_COVERAGE_THRESHOLD,
)

# --- paths -------------------------------------------------------------------
SCRATCH = os.environ["SCRATCH"]
NESTED_DIR = (f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-hrrr/stage1c"
              f"/inference-testing")
GLOBAL_DIR = (f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-only/stage1c"
              f"/inference-testing")
DEFAULT_OUT_DIR = f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"

TEST_START, TEST_END = "2024-02-01", "2025-01-31"
LEAD_HOURS = 24

# Reader configuration for the two model types (see the docstring / data.py).
VARS = ["u10", "v10", "t2m"]
LAM_INDEX = 407040
LCC_INFO = {"n_x": 848, "n_y": 480}
NESTED_KW = dict(model_type="nested-lam", vars_of_interest=VARS, load=False,
                 reshape_cell_to_2d=True, lcc_info=LCC_INFO, lam_index=LAM_INDEX)
GLOBAL_KW = dict(model_type="global", vars_of_interest=VARS, load=False,
                 reshape_cell_to_2d=True)


# --- cycles ------------------------------------------------------------------
def list_cycles():
    """(t0, nested_path, global_path) for every t0 in BOTH dirs, in-window.

    Filenames are ``<t0>.360h.nc`` with t0 like ``2024-02-01T06``.
    """
    def index(d):
        out = {}
        for p in glob.glob(f"{d}/*.nc"):
            stem = os.path.basename(p).split(".")[0]
            try:
                out[pd.Timestamp(stem)] = p
            except ValueError:
                continue
        return out

    nested, glob_ = index(NESTED_DIR), index(GLOBAL_DIR)
    lo = pd.Timestamp(TEST_START)
    hi = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)  # inclusive of last day
    common = sorted(t for t in (set(nested) & set(glob_)) if lo <= t < hi)
    return [(t, nested[t], glob_[t]) for t in common]


# --- per-cycle loading -------------------------------------------------------
def make_fields(ds):
    """{2m_temperature, 10m_wind_speed} on the reader's native grid."""
    out = xr.Dataset()
    out["2m_temperature"] = ds["t2m"]
    out["10m_wind_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    return out[FIELDS]


def load_lead(path, kw, valid_time):
    """Open lazily, select the fhr=LEAD_HOURS valid time, return native fields.

    load=False keeps the (28/20 GB) file lazy; selecting the single valid time
    before the regridder's .load() means only one lead x 3 vars is materialized.
    """
    ds = open_anemoi_inference_dataset(path, **kw)
    ds = ds.sel(time=valid_time, method="nearest", tolerance=pd.Timedelta("90m"))
    return make_fields(ds)


def coverage_mask(rg, grid_ds, grid_fn, s):
    """Per-station True where the station's bilinear stencil is inside the grid.

    xesmf locstream fills out-of-domain points with 0 (not NaN), so the Nested-LAM
    grid -- which is trimmed inside the HRRR domain -- would silently return 0 for
    the ~330 HRRR-domain stations beyond the LAM edge. Regridding an all-ones field
    gives the in-domain weight sum (~1 interior, 0 outside); we NaN the rest.
    """
    g = grid_fn(grid_ds)
    g = g.assign(ones=xr.ones_like(g["latitude"], dtype="float32"))
    cov = to_station(rg(g["ones"]), s)
    return cov >= VALID_COVERAGE_THRESHOLD


# --- metrics -----------------------------------------------------------------
def compute_metrics(nested, global_):
    """Per-station bias/rmse/variances over t0, stacked on a `field` dim."""
    keys = ("bias", "rmse", "var_diff", "var_nested", "var_global",
            "mean_nested", "mean_global", "count")
    metrics = {k: [] for k in keys}
    for f in FIELDS:
        d = nested[f] - global_[f]
        metrics["bias"].append(d.mean("t0"))
        metrics["rmse"].append(np.sqrt((d ** 2).mean("t0")))
        metrics["var_diff"].append(d.var("t0"))
        metrics["var_nested"].append(nested[f].var("t0"))
        metrics["var_global"].append(global_[f].var("t0"))
        metrics["mean_nested"].append(nested[f].mean("t0"))
        metrics["mean_global"].append(global_[f].mean("t0"))
        metrics["count"].append(d.notnull().sum("t0"))
    field = xr.DataArray(FIELDS, dims="field", name="field")
    return xr.Dataset({k: xr.concat(v, dim=field) for k, v in metrics.items()})


# --- main --------------------------------------------------------------------
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cycles = list_cycles()
    print(f"{len(cycles)} common cycles in [{TEST_START}, {TEST_END}]")

    # Same stations / HRRR-domain mask as the analysis study (1:1 with topo.nc).
    s = load_stations()
    s = s.isel(station=hrrr_valid_mask(s).values)
    print(f"{s.sizes['station']} stations inside the HRRR/LAM domain")

    # One regridder per model, built from the first cycle's (fixed) grid and
    # reused for every cycle. Nested = curvilinear LCC (y,x); Global = lat/lon.
    _, np0, gp0 = cycles[0]
    nested0 = open_anemoi_inference_dataset(np0, **NESTED_KW)
    nested_rg = make_regridder(nested0, hrrr_grid, s)
    global_rg = make_regridder(open_anemoi_inference_dataset(gp0, **GLOBAL_KW),
                               gfs_grid, s)

    # Stations beyond the (trimmed) Nested-LAM edge -> NaN (not xesmf's 0-fill).
    valid_nested = coverage_mask(nested_rg, nested0, hrrr_grid, s)
    print(f"{int(valid_nested.sum())} of {s.sizes['station']} stations inside the "
          f"Nested-LAM grid ({s.sizes['station'] - int(valid_nested.sum())} "
          f"outside -> NaN)")

    nested_list, global_list, t0s = [], [], []
    for i, (t0, np_, gp_) in enumerate(cycles, 1):
        valid = t0 + pd.Timedelta(hours=LEAD_HOURS)
        nf = regrid_fields(load_lead(np_, NESTED_KW, valid), hrrr_grid, nested_rg, s)
        gf = regrid_fields(load_lead(gp_, GLOBAL_KW, valid), gfs_grid, global_rg, s)
        nested_list.append(nf)
        global_list.append(gf)
        t0s.append(t0)
        print(f"  [{i}/{len(cycles)}] t0 {t0:%Y-%m-%dT%H} -> valid {valid:%Y-%m-%dT%H}")

    t0dim = xr.DataArray(pd.DatetimeIndex(t0s), dims="t0", name="t0")
    nested = xr.concat(nested_list, dim=t0dim).where(valid_nested)  # mask LAM edge
    global_ = xr.concat(global_list, dim=t0dim)

    metrics = compute_metrics(nested, global_)
    metrics.attrs.update(
        description="Nested-EAGLE minus Global-EAGLE 24 h forecast at obs stations",
        sign_convention="bias/diff = Nested - Global (positive => Nested larger)",
        lead_hours=LEAD_HOURS,
        test_period=f"{TEST_START}..{TEST_END}", n_cycles=len(t0s),
        nested_source=NESTED_DIR, global_source=GLOBAL_DIR,
        predictors="reuse gfs_vs_hrrr.topo.nc (orog/slope) and "
                   "gfs_vs_hrrr.roughness.nc (rough_diff); HRRR=Nested, GFS=Global",
        station_note="3585 stations (1:1 with topo/roughness); stations beyond the "
                     "Nested-LAM edge are NaN (count=0), not 0-filled")
    metrics["bias"].attrs["long_name"] = "mean_t(Nested - Global) @ fhr=24"
    metrics["rmse"].attrs["long_name"] = "sqrt(mean_t((Nested - Global)^2)) @ fhr=24"
    metrics["var_diff"].attrs["long_name"] = "var_t(Nested - Global); MSE = bias^2 + var_diff"
    metrics["var_nested"].attrs["long_name"] = "var_t(Nested)"
    metrics["var_global"].attrs["long_name"] = "var_t(Global)"
    mpath = os.path.join(out_dir, "nested_vs_global.fhr24.metrics.nc")
    metrics.to_netcdf(mpath)
    print(f"Wrote {mpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory")
    main(p.parse_args().out_dir)
