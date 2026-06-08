"""
Preprocess orography for the spatial-obs-metrics error analysis.

Goal
----
The spatial obs-metrics workflow (eagle.tools.spatial_obs_metrics) produces
error/bias/mae/count per *station* (the obs location, rounded). To explain those
errors in terms of topography we need, at each station location:

  1. ``orography``  -- the surface elevation (m).
  2. ``grad_mag``   -- the amplitude of the local topographic gradient
                       |grad(h)|, a dimensionless slope (m rise / m run).

We compute |grad(h)| *on the source's native grid* (so the slope is defined at
that model's resolution, not an arbitrary downstream grid), then bilinear-sample
elevation and slope to the station locations (xesmf locstream). Two source grid
layouts are supported:

  * curvilinear Lambert-conformal (HRRR): ``orog(y, x)`` with 2-D lat/lon. Slope
    via centred differences along y/x with the median cell spacing in metres.
  * regular lat/lon (GFS): ``orog(latitude, longitude)`` 1-D. Slope via degree
    derivatives with the cos(lat) metric.

Use ``--orog-source`` to pick the DEM (HRRR 6 km / 3 km, GFS, ...). This lets you
test whether the error tracks the *true* terrain or a given model's smoothed
*representation* of it. The ``valid`` (in-HRRR-domain) mask is taken from a
separate ``--domain-source`` (default HRRR 6 km) so the analysis stays on the
same stations regardless of which orography is used.

Outputs ``orography_and_gradient.nc`` (dims: station) next to the error files.
"""
import argparse
import os

import numpy as np
import xarray as xr
import xesmf

# --- paths -------------------------------------------------------------------
SCRATCH = os.environ["SCRATCH"]
# 6 km HRRR orography the model was trained on (conservatively regridded from 3 km).
DEFAULT_OROG_SOURCE = f"{SCRATCH}/nested-eagle/case-studies/la-fires/hrrr.forecasts.zarr"
# Defines the HRRR domain for the in-domain `valid` mask (kept fixed across DEMs).
DEFAULT_DOMAIN_SOURCE = DEFAULT_OROG_SOURCE
DEFAULT_METRICS_DIR = (
    f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-hrrr/stage1c"
    f"/inference-testing/spatial-obs-metrics"
)
DEFAULT_MODEL_TAG = "nested-lam"

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius

# A station whose bilinear stencil is not (nearly) fully inside the domain is
# treated as outside it. Interior points get weight 1; the boundary ring < 1.
VALID_COVERAGE_THRESHOLD = 0.99


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (m) between two arrays of lat/lon points (degrees)."""
    p = np.pi / 180.0
    dphi = (lat2 - lat1) * p
    dl = (lon2 - lon1) * p
    a = (np.sin(dphi / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_orography(source):
    """Load static ``orog`` (first init time) from a zarr; sort regular grids."""
    o = xr.open_zarr(source)["orog"]
    for tdim in ("t0", "time"):
        if tdim in o.dims:  # orography ~static over the window; take the first
            o = o.isel({tdim: 0})
            break
    o = o.load()
    if o["latitude"].ndim == 1:  # regular grid: ensure ascending lat for xesmf/diff
        o = o.sortby("latitude")
    return o


def gradient_amplitude(orog):
    """|grad(h)| at the source's native resolution (dimensionless slope).

    Returns the slope DataArray and a short text description of the scale.
    """
    if orog["latitude"].ndim == 2:  # curvilinear (y, x): centred diffs along axes
        lat = orog["latitude"].values
        lon = orog["longitude"].values % 360.0
        dy = float(np.median(haversine(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])))
        dx = float(np.median(haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])))
        gy, gx = np.gradient(orog.values, dy, dx)
        grad = xr.DataArray(np.sqrt(gx**2 + gy**2), dims=orog.dims, coords=orog.coords)
        return grad, f"curvilinear native grid (dy={dy:.0f} m, dx={dx:.0f} m)"
    # regular lat/lon grid: degree derivatives with the cos(lat) metric
    deg2m = EARTH_RADIUS_M * np.deg2rad(1.0)
    coslat = np.cos(np.deg2rad(orog["latitude"]))
    gy = orog.differentiate("latitude") / deg2m
    gx = orog.differentiate("longitude") / (deg2m * coslat)
    grad = np.sqrt(gx**2 + gy**2)
    dlat = abs(float(np.diff(orog["latitude"].values).mean()))
    return grad, f"regular {dlat:.3f} deg grid (~{dlat * deg2m / 1000:.0f} km)"


def build_source_grid(orog, grad):
    """xesmf-ready source carrying elevation and slope (curvilinear or regular)."""
    if orog["latitude"].ndim == 2:  # curvilinear
        return xr.Dataset(
            {"orog": (("y", "x"), orog.values), "grad": (("y", "x"), grad.values)},
            coords={"lat": (("y", "x"), orog["latitude"].values),
                    "lon": (("y", "x"), orog["longitude"].values % 360.0)},
        )
    return xr.Dataset(  # regular lat/lon
        {"orog": (("latitude", "longitude"), orog.values),
         "grad": (("latitude", "longitude"), grad.values)},
        coords={"lat": (("latitude",), orog["latitude"].values),
                "lon": (("longitude",), orog["longitude"].values % 360.0)},
    )


def _sample_to_stations(ds_in, field, err):
    """Bilinear-sample a source field to the error file's station locations."""
    ds_out = xr.Dataset(coords={
        "lat": ("station", err["latitude"].values),
        "lon": ("station", err["longitude"].values),
    })
    rg = xesmf.Regridder(ds_in, ds_out, method="bilinear", locstream_out=True)
    return rg(ds_in[field])


def main(orog_source, error_file, out_file, domain_source):
    err = xr.open_dataset(error_file)

    orog = load_orography(orog_source)
    grad, scale = gradient_amplitude(orog)
    print(f"Orography source: {orog_source}\n  slope scale: {scale}")
    ds_in = build_source_grid(orog, grad)
    orography = _sample_to_stations(ds_in, "orog", err)
    grad_st = _sample_to_stations(ds_in, "grad", err)

    # in-domain mask from the (fixed) HRRR domain, independent of the orog source:
    # bilinear weights sum to ~1 inside the domain and < 1 where the stencil pokes
    # outside, so coverage of an all-ones field gives the in-domain fraction.
    dom = load_orography(domain_source)
    dom_in = build_source_grid(dom, xr.zeros_like(dom))
    coverage = _sample_to_stations(dom_in.assign(ones=xr.ones_like(dom_in["orog"])),
                                   "ones", err)
    valid = coverage >= VALID_COVERAGE_THRESHOLD

    def _to_station(da, name):
        da = da.rename({"locations": "station"}) if "locations" in da.dims else da
        da = da.assign_coords(
            station=err["station"], latitude=err["latitude"], longitude=err["longitude"])
        da.name = name
        return da

    orography = _to_station(orography, "orography")
    orography.attrs.update(units="m", long_name="surface elevation at station")
    grad_st = _to_station(grad_st, "grad_mag")
    grad_st.attrs.update(
        units="m m-1",
        long_name="amplitude of local topographic gradient |grad(h)| (dimensionless slope)",
        note=f"computed on {scale}",
    )
    valid = _to_station(valid.astype("int8"), "valid")
    valid.attrs.update(
        long_name="1 where the station's bilinear stencil is inside the HRRR domain",
        coverage_threshold=VALID_COVERAGE_THRESHOLD, domain_source=domain_source)

    out = xr.Dataset({"orography": orography, "grad_mag": grad_st, "valid": valid})
    out.attrs.update(source_orography=orog_source, station_source=error_file,
                     domain_source=domain_source, slope_scale=scale)
    out.to_netcdf(out_file)
    print(f"Wrote {out_file}  ({out.sizes['station']} stations, {int(valid.sum())} valid)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orog-source", default=DEFAULT_OROG_SOURCE,
                   help="zarr with an 'orog' field (curvilinear or regular). "
                        "Default: 6 km HRRR")
    p.add_argument("--domain-source", default=DEFAULT_DOMAIN_SOURCE,
                   help="zarr defining the in-domain 'valid' mask (default: 6 km HRRR); "
                        "keep fixed across orography sources for a fair comparison")
    p.add_argument("--metrics-dir", default=DEFAULT_METRICS_DIR,
                   help="dir holding the spatial.*.convobs.<tag>.nc error files")
    p.add_argument("--model-tag", default=DEFAULT_MODEL_TAG,
                   help="filename tag, e.g. 'nested-lam' or 'global'")
    p.add_argument("--out", default=None,
                   help="output netcdf (default: <metrics-dir>/orography_and_gradient.nc)")
    args = p.parse_args()
    error_file = f"{args.metrics_dir}/spatial.rmse.convobs.{args.model_tag}.nc"
    out_file = args.out or f"{args.metrics_dir}/orography_and_gradient.nc"
    main(args.orog_source, error_file, out_file, args.domain_source)
