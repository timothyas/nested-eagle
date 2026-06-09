"""
Compare GFS vs HRRR representations of 2 m temperature and 10 m wind speed at
conventional-obs station locations, over the test year.

Motivation
----------
We have been evaluating GFS-vs-obs and HRRR-vs-obs separately and the analysis
got convoluted. Since both are scored against the same obs, the *difference*
between the two analyses (initial conditions) is directly informative: it is the
part of the obs-space error gap that is purely a GFS<->HRRR representation
difference, independent of the obs themselves. We evaluate it at the obs station
locations (a meaningful, terrain-following sample of the domain) rather than on a
full grid.

For each station and each field we interpolate both analyses (fhr=0) to the
station and, over the test-year cycles, compute:

  - bias       : mean_t(HRRR - GFS)              (sign: + => HRRR larger)
  - rmse       : sqrt(mean_t((HRRR - GFS)^2))
  - var_diff   : var_t(HRRR - GFS)               (error variance; MSE = bias^2 + var_diff)
  - var_gfs    : var_t(GFS)                       (each model's own temporal variability)
  - var_hrrr   : var_t(HRRR)
  - mean_gfs   : mean_t(GFS)                      (context for the bias)
  - mean_hrrr  : mean_t(HRRR)
  - count      : number of cycles contributing

Wind is treated as the scalar 10 m wind speed sqrt(u^2 + v^2), computed on each
model's native grid before interpolation (matching the prior topo analysis).

Separately we record the static terrain difference each model "sees" at every
station: surface elevation and the amplitude of the local topographic gradient
|grad h| (the native-grid slope), for GFS and HRRR and their HRRR-minus-GFS
difference. This lets later analysis relate the representation differences above
to how differently the two models resolve the terrain.

Outputs (to OUT_DIR):
  gfs_vs_hrrr.metrics.nc  -- per-station bias/rmse/variances (dims: field, station)
  gfs_vs_hrrr.topo.nc     -- per-station orography & slope, per model and diff
"""
import argparse
import os

import numpy as np
import xarray as xr
import xesmf

# --- paths -------------------------------------------------------------------
SCRATCH = os.environ["SCRATCH"]
GFS_FORECASTS = f"{SCRATCH}/nested-eagle/0.25deg-06km/baselines/gfs-forecasts-vs-gfs-analysis/gfs.forecasts.zarr"
HRRR_FORECASTS = f"{SCRATCH}/nested-eagle/0.25deg-06km/baselines/hrrr-forecasts-vs-hrrr-analysis/hrrr.forecasts.zarr"
# These carry the static `orog` field on the *same* grids as the forecast zarrs.
GFS_OROG = f"{SCRATCH}/nested-eagle/case-studies/la-fires/gfs.forecasts.zarr"
HRRR_OROG = f"{SCRATCH}/nested-eagle/case-studies/la-fires/hrrr.forecasts.zarr"
STATION_FILE = (
    f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-hrrr/stage1c"
    f"/inference-testing/spatial-obs-metrics/spatial.count.convobs.nested-lam.nc"
)
DEFAULT_OUT_DIR = f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"

TEST_START, TEST_END = "2024-02-01", "2025-01-31"
FIELDS = ["2m_temperature", "10m_wind_speed"]

EARTH_RADIUS_M = 6_371_000.0
# A station whose bilinear stencil is not (nearly) fully inside the HRRR domain
# is treated as outside it (and dropped). Interior points get weight 1.
VALID_COVERAGE_THRESHOLD = 0.99


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (m) between arrays of lat/lon points (degrees)."""
    p = np.pi / 180.0
    dphi = (lat2 - lat1) * p
    dl = (lon2 - lon1) * p
    a = (np.sin(dphi / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# --- stations ----------------------------------------------------------------
def load_stations():
    """Finite-location stations from the obs-metrics file (dims: station)."""
    s = xr.open_dataset(STATION_FILE)[["latitude", "longitude"]]
    finite = np.isfinite(s["latitude"]) & np.isfinite(s["longitude"])
    s = s.isel(station=finite.values)
    return s


def station_out(s):
    """xesmf locstream-out target carrying the station lat/lon."""
    return xr.Dataset(coords={
        "lat": ("station", s["latitude"].values),
        "lon": ("station", s["longitude"].values % 360.0),
    })


# --- source grids ------------------------------------------------------------
def gfs_grid(o):
    """Attach xesmf lat/lon coords to a regular-grid (latitude, longitude) array
    or dataset; sort latitude ascending as xesmf/diff prefer."""
    o = o.sortby("latitude")
    return o.assign_coords(lat=o["latitude"], lon=o["longitude"] % 360.0)


def hrrr_grid(o):
    """Attach xesmf lat/lon coords to a curvilinear (y, x) array or dataset."""
    return o.assign_coords(lat=o["latitude"], lon=o["longitude"] % 360.0)


def to_station(da, s):
    """Standardize a regridded DataArray onto the station dim with coords."""
    if "locations" in da.dims:
        da = da.rename({"locations": "station"})
    return da.assign_coords(station=s["station"], latitude=s["latitude"],
                            longitude=s["longitude"])


# --- forecast loading --------------------------------------------------------
def load_analysis(zarr, native_dims):
    """fhr=0 analyses over the test period: 2 m T and 10 m wind speed.

    Returns a Dataset with the two fields on the source's native horizontal dims,
    plus a `t0` dim. Wind speed is formed on the native grid before interpolation.
    """
    ds = xr.open_zarr(zarr).sel(fhr=0).sel(t0=slice(TEST_START, TEST_END))
    out = xr.Dataset()
    out["2m_temperature"] = ds["t2m"]
    out["10m_wind_speed"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    return out[FIELDS]


def make_regridder(grid_ds, grid_fn, s):
    """Build the bilinear station regridder for a model's native grid.

    The weights depend only on the source grid and the (fixed) station
    locations, so one regridder per model is reused across all cycles AND for
    the static terrain sampling below.
    """
    return xesmf.Regridder(grid_fn(grid_ds), station_out(s), method="bilinear",
                           locstream_out=True)


def regrid_fields(ds_native, grid_fn, rg, s):
    """Interpolate native-grid analysis fields to stations with a prebuilt rg.

    Returns a Dataset (field vars) with dims (t0, station), loaded into memory.
    A single rg() call broadcasts over all t0, so the regridder is reused across
    every cycle rather than rebuilt per cycle.
    """
    ds_in = grid_fn(ds_native)
    out = xr.Dataset()
    for f in FIELDS:
        out[f] = to_station(rg(ds_in[f]), s)
    return out.load()


# --- topography --------------------------------------------------------------
def load_orog(source):
    """Static `orog` (first cycle) from a forecast zarr."""
    o = xr.open_zarr(source)["orog"]
    for tdim in ("t0", "time"):
        if tdim in o.dims:  # orography ~static; take first cycle, drop the
            o = o.isel({tdim: 0}, drop=True)  # scalar t0 coord (differs per source)
            break
    return o.load()


def slope_amplitude(orog):
    """|grad h| at the source's native resolution (dimensionless slope)."""
    if orog["latitude"].ndim == 2:  # curvilinear (y, x): centred diffs on axes
        lat = orog["latitude"].values
        lon = orog["longitude"].values % 360.0
        dy = float(np.median(haversine(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])))
        dx = float(np.median(haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])))
        gy, gx = np.gradient(orog.values, dy, dx)
        return xr.DataArray(np.sqrt(gx ** 2 + gy ** 2), dims=orog.dims,
                            coords=orog.coords)
    # regular lat/lon: degree derivatives with the cos(lat) metric
    deg2m = EARTH_RADIUS_M * np.deg2rad(1.0)
    coslat = np.cos(np.deg2rad(orog["latitude"]))
    gy = orog.differentiate("latitude") / deg2m
    gx = orog.differentiate("longitude") / (deg2m * coslat)
    return np.sqrt(gx ** 2 + gy ** 2)


def sample_topo(source, grid_fn, rg, s):
    """Bilinear-sample (orography, slope) to stations, reusing the model's rg.

    `source` is the DEM zarr (same native grid the rg was built for); the slope
    is the |grad h| amplitude on that native grid.
    """
    orog = load_orog(source)
    if orog["latitude"].ndim == 1:
        orog = orog.sortby("latitude")
    slope = slope_amplitude(orog)
    ds_in = grid_fn(xr.Dataset({"orog": orog, "slope": slope}))
    return to_station(rg(ds_in["orog"]), s), to_station(rg(ds_in["slope"]), s)


def hrrr_valid_mask(s):
    """1 where a station's HRRR bilinear stencil lies inside the HRRR domain.

    Coverage of an all-ones HRRR field equals the in-domain bilinear-weight sum,
    which is ~1 in the interior and < 1 where the stencil pokes outside.
    """
    dom = hrrr_grid(load_orog(HRRR_OROG).to_dataset(name="orog"))
    dom["ones"] = xr.ones_like(dom["orog"])
    rg = xesmf.Regridder(dom, station_out(s), method="bilinear",
                         locstream_out=True)
    cov = to_station(rg(dom["ones"]), s)
    return cov >= VALID_COVERAGE_THRESHOLD


# --- metrics -----------------------------------------------------------------
def compute_metrics(gfs, hrrr):
    """Per-station bias/rmse/variances over t0, stacked on a `field` dim."""
    metrics = {k: [] for k in ("bias", "rmse", "var_diff", "var_gfs", "var_hrrr",
                               "mean_gfs", "mean_hrrr", "count")}
    for f in FIELDS:
        d = hrrr[f] - gfs[f]
        metrics["bias"].append(d.mean("t0"))
        metrics["rmse"].append(np.sqrt((d ** 2).mean("t0")))
        metrics["var_diff"].append(d.var("t0"))
        metrics["var_gfs"].append(gfs[f].var("t0"))
        metrics["var_hrrr"].append(hrrr[f].var("t0"))
        metrics["mean_gfs"].append(gfs[f].mean("t0"))
        metrics["mean_hrrr"].append(hrrr[f].mean("t0"))
        metrics["count"].append(d.notnull().sum("t0"))
    field = xr.DataArray(FIELDS, dims="field", name="field")
    out = xr.Dataset({k: xr.concat(v, dim=field) for k, v in metrics.items()})
    return out


# --- main --------------------------------------------------------------------
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    s = load_stations()
    print(f"{s.sizes['station']} finite-location stations")

    valid = hrrr_valid_mask(s)
    s = s.isel(station=valid.values)
    print(f"{s.sizes['station']} stations inside the HRRR domain")

    # One regridder per model, built once from its native grid and reused for
    # every cycle and for the terrain sampling.
    gfs_rg = make_regridder(load_orog(GFS_OROG).to_dataset(name="orog"), gfs_grid, s)
    hrrr_rg = make_regridder(load_orog(HRRR_OROG).to_dataset(name="orog"), hrrr_grid, s)

    print("Interpolating GFS analyses to stations ...")
    gfs = regrid_fields(load_analysis(GFS_FORECASTS, ("latitude", "longitude")),
                        gfs_grid, gfs_rg, s)
    print(f"  {gfs.sizes['t0']} cycles")
    print("Interpolating HRRR analyses to stations ...")
    hrrr = regrid_fields(load_analysis(HRRR_FORECASTS, ("y", "x")), hrrr_grid,
                         hrrr_rg, s)

    metrics = compute_metrics(gfs, hrrr)
    metrics.attrs.update(
        description="HRRR-minus-GFS analysis (fhr=0) comparison at obs stations",
        sign_convention="bias/diff = HRRR - GFS (positive => HRRR larger)",
        test_period=f"{TEST_START}..{TEST_END}", n_cycles=int(gfs.sizes["t0"]),
        gfs_source=GFS_FORECASTS, hrrr_source=HRRR_FORECASTS)
    metrics["bias"].attrs["long_name"] = "mean_t(HRRR - GFS)"
    metrics["rmse"].attrs["long_name"] = "sqrt(mean_t((HRRR - GFS)^2))"
    metrics["var_diff"].attrs["long_name"] = "var_t(HRRR - GFS); MSE = bias^2 + var_diff"
    metrics["var_gfs"].attrs["long_name"] = "var_t(GFS)"
    metrics["var_hrrr"].attrs["long_name"] = "var_t(HRRR)"
    mpath = os.path.join(out_dir, "gfs_vs_hrrr.metrics.nc")
    metrics.to_netcdf(mpath)
    print(f"Wrote {mpath}")

    # Per-cycle difference time series d(field, t0, station) -- the reusable
    # primitive behind var_diff (var_t(d) == var_diff), needed for the temporal
    # (season/diurnal) variance decomposition. Small: 2 x ~293 x ~3585 floats.
    field = xr.DataArray(FIELDS, dims="field", name="field")
    diffs = xr.concat([hrrr[f] - gfs[f] for f in FIELDS], dim=field).to_dataset(name="diff")
    diffs.attrs.update(
        description="per-cycle HRRR - GFS analysis (fhr=0) difference at obs stations",
        sign_convention="diff = HRRR - GFS", test_period=f"{TEST_START}..{TEST_END}")
    diffs["diff"].attrs["long_name"] = "HRRR - GFS at fhr=0 (valid time = t0)"
    dpath = os.path.join(out_dir, "gfs_vs_hrrr.diffs.nc")
    diffs.to_netcdf(dpath)
    print(f"Wrote {dpath}")

    print("Sampling terrain to stations ...")
    orog_g, slope_g = sample_topo(GFS_OROG, gfs_grid, gfs_rg, s)
    orog_h, slope_h = sample_topo(HRRR_OROG, hrrr_grid, hrrr_rg, s)
    topo = xr.Dataset({
        "orog_gfs": orog_g, "orog_hrrr": orog_h, "orog_diff": orog_h - orog_g,
        "slope_gfs": slope_g, "slope_hrrr": slope_h, "slope_diff": slope_h - slope_g,
    })
    topo["orog_gfs"].attrs.update(units="m", long_name="GFS surface elevation")
    topo["orog_hrrr"].attrs.update(units="m", long_name="HRRR surface elevation")
    topo["orog_diff"].attrs.update(units="m", long_name="HRRR - GFS surface elevation")
    for k in ("slope_gfs", "slope_hrrr", "slope_diff"):
        topo[k].attrs.update(units="m m-1", long_name="|grad h| native-grid slope")
    topo.attrs.update(
        sign_convention="diff = HRRR - GFS",
        gfs_orog_source=GFS_OROG, hrrr_orog_source=HRRR_OROG)
    tpath = os.path.join(out_dir, "gfs_vs_hrrr.topo.nc")
    topo.to_netcdf(tpath)
    print(f"Wrote {tpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory")
    main(p.parse_args().out_dir)
