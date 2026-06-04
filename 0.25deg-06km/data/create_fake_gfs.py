from copy import deepcopy
import logging

import numpy as np
import pandas as pd
import xarray as xr

from anemoi.datasets import open_dataset

from eagle.tools.data import open_anemoi_dataset
from eagle.tools.nested import regrid_nested_to_latlon

import ufs2arco.targets
from ufs2arco.sources import Source
from ufs2arco.log import setup_simple_log
from ufs2arco.mpi import MPITopology, SerialTopology

logger = logging.getLogger("ufs2arco")

_start = "2023-02-01T00"
_end = "2025-02-15T18"

_global_dataset = "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/testing/gfs.zarr"
_lam_dataset = "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/testing/hrrr.zarr"
_fake_gfs_path =  "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/testing/gfs.with.hrrr.singlerez.zarr"
_target_grid_path = "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/mesh-gen/csmswt-trim25/global_quarter_degree_with_mask.nc"
_regridder_weights_filename = "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/mesh-gen/csmswt-trim25/conservative_lam_to_global.nc"

_n_x = 848
_n_y = 480
_lam_index = 407040
_min_distance_km = 6
_trim_edge = (25, 24, 25, 26)


# Note: this is used to avoid dropping the forcings and static variables
_vars_of_interest = [
    "gh",
    "u",
    "v",
    "w",
    "t",
    "q",
    "sp",
    "u10",
    "v10",
    "t2m",
    "t_surface",
    "sh2",
    "u80",
    "v80",
    "accum_tp",
    "lsm",
    "orog",
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
    "insolation",
]

class PassiveSource(Source):
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    static_vars = ("lsm", "orog")

    def __init__(self, xds, time):
        variables = list(xds.data_vars)
        levels = list(xds.level.values)
        self.available_variables = tuple(variables)
        self.available_levels = tuple(levels)

        self.time = time
        super().__init__(
            variables=variables,
            levels=levels,
        )


def open_regrid_reformat(date):
    ads = open_anemoi_dataset(
        cutout=[
            {
                "dataset": _lam_dataset,
                "trim_edge": list(_trim_edge),
            },
            _global_dataset,
        ],
        adjust="all",
        min_distance_km=_min_distance_km,
        model_type="nested",
        t0=date,
        tf=date,
        vars_of_interest=_vars_of_interest,
    )

    if "member" in ads:
        ads = ads.isel(member=0)
        ads = ads.drop_vars("member")

    xds = regrid_nested_to_latlon(
        ads,
        lam_index=_lam_index,
        lcc_info={"n_x": _n_x, "n_y": _n_y},
        horizontal_regrid_kwargs={
            "target_grid_path": _target_grid_path,
            "regridder_kwargs": {
                "method": "conservative_normed",
                "reuse_weights": True,
                "filename": _regridder_weights_filename,
            },
        },
    )
    return xds

def get_source_and_target(xds, time, variable_order):
    source = PassiveSource(xds, time=time)
    target = ufs2arco.targets.Anemoi(
        source=source,
        store_path=_fake_gfs_path,
        sort_channels_by_levels=True,
        chunks={
            "time": 1,
            "variable": -1,
            "ensemble": 1,
            "cell": -1,
        },
        variable_order=variable_order,
    )


    return source, target

def get_variable_order():
    reference = open_dataset(_global_dataset)
    return tuple(deepcopy(reference.variables))

def find_my_region(xds, itime):
    region = {k: slice(None, None) for k in xds.dims}
    region["time"] = slice(itime, itime+1)
    return region

if __name__ == "__main__":

    try:
        topo = MPITopology(log_dir="logs-fake-gfs")
    except AssertionError:
        topo = SerialTopology()
        setup_simple_log()

    time = pd.date_range(_start, _end, freq="6h")
    variable_order = get_variable_order()

    # Rank 0 creates the zarr container and writes the first time step
    date0 = time[0].strftime("%Y-%m-%dT%H")
    xds0 = open_regrid_reformat(date=date0)
    source, target = get_source_and_target(xds0, time=time, variable_order=variable_order)

    if topo.is_root:
        logger.info("Making container ...")
        cds = target.container_maker(xds0)
        cds.to_zarr(target.store_path, compute=False)
        logger.info("Wrote container...")

        xds0 = target.apply_transforms_to_sample(xds0)
        region0 = find_my_region(xds0, 0)
        xds0.to_zarr(_fake_gfs_path, region=region0)
        logger.info(f"Wrote {date0}")

    topo.barrier()

    # All ranks process remaining time steps in parallel
    remaining_time = time[1:]
    n_times = len(remaining_time)
    n_batches = int(np.ceil(n_times / topo.size))

    for batch_idx in range(n_batches):
        itime = batch_idx * topo.size + topo.rank
        if itime >= n_times:
            break
        t = remaining_time[itime]
        actual_itime = itime + 1  # offset: time[0] was handled by root
        date = t.strftime("%Y-%m-%dT%H")
        xds = open_regrid_reformat(date=date)
        xds = target.apply_transforms_to_sample(xds)
        region = find_my_region(xds, actual_itime)
        xds.to_zarr(_fake_gfs_path, region=region)
        logger.info(f"Wrote {date}")

    topo.barrier()
    logger.info(f"Done moving data")

    target.finalize(topo)
