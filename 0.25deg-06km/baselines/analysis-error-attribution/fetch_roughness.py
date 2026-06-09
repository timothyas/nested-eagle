"""
Pull true surface roughness (SFCR, aerodynamic z0) for GFS and HRRR from the
public NOAA GRIB archives and sample it to the obs stations, to test whether the
10 m wind difference (HRRR - GFS) is a roughness-representation effect.

The forecast/native zarrs carry no roughness, so we byte-range the single SFCR
message out of each cycle's GRIB via its .idx (no full-file download):
  GFS : s3://noaa-gfs-bdp-pds/gfs.<d>/<HH>/atmos/gfs.t<HH>z.pgrb2.0p25.f000
  HRRR: s3://noaa-hrrr-bdp-pds/hrrr.<d>/conus/hrrr.t<HH>z.wrfsfcf00.grib2

Roughness is quasi-static but varies seasonally (vegetation green-up, snow), and
the wind bias is a full-year time mean, so we sample one cycle per month (the
test-year t0 nearest each 15th) and average log10(z0). We compare in log space
because z0 spans orders of magnitude and the two models define it differently
(GFS z0 up to ~2.6 m, HRRR up to ~0.8 m).

Output: gfs_vs_hrrr.roughness.nc with z0_gfs, z0_hrrr (geometric-mean m),
log10_z0_gfs/hrrr, and rough_diff = log10_z0_hrrr - log10_z0_gfs (HRRR rougher>0).
"""
import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import xesmf
import s3fs

from compare_gfs_hrrr import station_out, to_station

SCRATCH = os.environ["SCRATCH"]
DATA = f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"
METRICS = f"{DATA}/gfs_vs_hrrr.metrics.nc"
GFS_FORECASTS = f"{SCRATCH}/nested-eagle/0.25deg-06km/baselines/gfs-forecasts-vs-gfs-analysis/gfs.forecasts.zarr"
TEST_START, TEST_END = "2024-02-01", "2025-01-31"
Z0_FLOOR = 1e-4  # m; floor before log10 (water/ice z0 ~ 1e-4..1e-3, GFS has exact 0s)

fs = s3fs.S3FileSystem(anon=True)


def fetch_sfcr(path):
    """Byte-range the SFCR message from a GRIB on S3 and decode it (loaded)."""
    idx = fs.cat(path + ".idx").decode().splitlines()
    for i, line in enumerate(idx):
        if ":SFCR:surface:anl:" in line:
            start = int(line.split(":")[1])
            end = int(idx[i + 1].split(":")[1]) if i + 1 < len(idx) else None
            raw = fs.cat_file(path, start=start, end=end)
            tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
            tmp.write(raw)
            tmp.close()
            ds = xr.open_dataset(tmp.name, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""}).load()
            os.unlink(tmp.name)
            return ds[list(ds.data_vars)[0]]  # the lone 'fsr' field
    raise RuntimeError(f"SFCR not found in {path}.idx")


def gfs_path(d, HH):
    return f"noaa-gfs-bdp-pds/gfs.{d}/{HH}/atmos/gfs.t{HH}z.pgrb2.0p25.f000"


def hrrr_path(d, HH):
    return f"noaa-hrrr-bdp-pds/hrrr.{d}/conus/hrrr.t{HH}z.wrfsfcf00.grib2"


def log_z0_grid(da):
    """Floor z0 and take log10, keeping native lat/lon coords for xesmf."""
    lz = np.log10(da.clip(min=Z0_FLOOR))
    return lz.assign_coords(lat=da["latitude"], lon=da["longitude"] % 360.0)


def make_regridder(grid, s):
    """Bilinear regridder from a (regular or curvilinear) z0 grid to stations."""
    if grid["latitude"].ndim == 1:
        grid = grid.sortby("latitude")
    return xesmf.Regridder(grid.to_dataset(name="lz"), station_out(s),
                           method="bilinear", locstream_out=True), grid


def monthly_cycles():
    """One test-year t0 per month: the cycle nearest that month's 15th."""
    t0 = xr.open_zarr(GFS_FORECASTS).sel(t0=slice(TEST_START, TEST_END)).t0
    t0 = pd.to_datetime(t0.values)
    out = []
    for (yr, mo), grp in pd.Series(t0).groupby([t0.year, t0.month]):
        target = pd.Timestamp(yr, mo, 15)
        out.append(grp.iloc[(grp - target).abs().values.argmin()])
    return sorted(out)


def main():
    s = xr.open_dataset(METRICS)[["latitude", "longitude"]]
    print(f"{s.sizes['station']} stations")
    cycles = monthly_cycles()
    print(f"{len(cycles)} monthly cycles: "
          f"{cycles[0]:%Y-%m-%d %Hz} .. {cycles[-1]:%Y-%m-%d %Hz}")

    g_rg = h_rg = None
    g_acc, h_acc = [], []
    for c in cycles:
        d, HH = f"{c:%Y%m%d}", f"{c:%H}"
        g = log_z0_grid(fetch_sfcr(gfs_path(d, HH)))
        h = log_z0_grid(fetch_sfcr(hrrr_path(d, HH)))
        if g_rg is None:  # build the (static-grid) regridders once
            g_rg, g_sorted_ref = make_regridder(g, s)
            h_rg, _ = make_regridder(h, s)
        if g["latitude"].ndim == 1:
            g = g.sortby("latitude")
        g_acc.append(to_station(g_rg(g), s))
        h_acc.append(to_station(h_rg(h), s))
        print(f"  {c:%Y-%m-%d %Hz}: GFS/HRRR sampled")

    log_g = xr.concat(g_acc, dim="cycle").mean("cycle")
    log_h = xr.concat(h_acc, dim="cycle").mean("cycle")
    out = xr.Dataset({
        "log10_z0_gfs": log_g, "log10_z0_hrrr": log_h,
        "z0_gfs": 10 ** log_g, "z0_hrrr": 10 ** log_h,
        "rough_diff": (log_h - log_g),
    })
    out["rough_diff"].attrs.update(
        long_name="log10(z0_hrrr) - log10(z0_gfs); >0 => HRRR rougher",
        note="dimensionless log10 ratio of aerodynamic roughness lengths")
    for k in ("z0_gfs", "z0_hrrr"):
        out[k].attrs.update(units="m", long_name="geometric-mean surface roughness z0")
    out.attrs.update(
        source="NOAA GFS/HRRR GRIB SFCR (surface roughness), fhr=0",
        sampling=f"{len(cycles)} monthly cycles {TEST_START}..{TEST_END}",
        z0_floor_m=Z0_FLOOR)
    path = f"{DATA}/gfs_vs_hrrr.roughness.nc"
    out.to_netcdf(path)
    print(f"Wrote {path}")
    print(f"  rough_diff mean {float(out.rough_diff.mean()):+.2f} "
          f"(median {float(out.rough_diff.median()):+.2f}); "
          f"z0_gfs median {float(out.z0_gfs.median()):.3f} m, "
          f"z0_hrrr median {float(out.z0_hrrr.median()):.3f} m")


if __name__ == "__main__":
    main()
