import os
import yaml
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Optional, Tuple, Union

import xarray as xr
import numpy as np
import pandas as pd

import xesmf as xe
from ufs2arco.utils import convert_anemoi_inference_dataset
from ufs2arco.transforms.horizontal_regrid import get_bounds

from eagle.log import setup_simple_log
from eagle.utils import open_yaml_config

logger = logging.getLogger("eagle")

def open_raw_inference(path_to_raw_inference: str) -> xr.Dataset:
    """
    Open one Anemoi-Inference run.

    Args:
        path_to_raw_inference (str): Path to an Anemoi-Inference run.

    Returns:
        xr.Dataset: One initialization of an Anemoi-Inference run.
    """
    xds = xr.open_dataset(path_to_raw_inference, chunks="auto")
    return convert_anemoi_inference_dataset(xds)


def mask_values(
    area_to_return: str, ds_nested: xr.Dataset, lam_index: int
) -> xr.Dataset:
    """
    Mask dataset values based on LAM coordinates.

    Args:
        area_to_return (str): Either "lam" or "global" to specify which area to return.
        ds (xr.Dataset): Input nested dataset to mask.
        lam_index (int): Index where nested ds transitions from LAM->global.

    Returns:
        xr.Dataset: Masked dataset containing either LAM only ds or global only (lam missing) ds.
    """
    if area_to_return == "lam":
        return ds_nested.isel(cell=slice(lam_index))
    elif area_to_return == "global":
        return ds_nested.isel(cell=slice(lam_index, None))
    else:
        raise ValueError("area_to_return must be either 'lam' or 'global'")


def create_2D_grid(
    ds: xr.Dataset,
    lcc_info: dict = None,
) -> xr.Dataset:
    """
    Reshape dataset from 1D 'cell' dimension to 2D latitude and longitude.

    Args:
        ds (xr.Dataset): Anemoi dataset with a flattened "cell" dimension.
        lcc_info (dict): Necesary info about LCC configuation.

    Returns:
        xr.Dataset: Dataset with shape (time, latitude, longitude).
    """
    ds_to_reshape = ds.copy()
    logger.info(f"ds_to_reshape:\n{ds_to_reshape}")

    if lcc_info:
        lat_length = lcc_info["lat_length"]
        lon_length = lcc_info["lon_length"]

        time_length = len(ds_to_reshape["time"].values)

        ds_to_reshape["x"] = np.arange(0, lon_length)
        ds_to_reshape["y"] = np.arange(0, lat_length)

        lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
        lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))

        data_vars = {}
        dims = {"time": time_length, "level": len(ds_to_reshape["level"]), "y": lat_length, "x": lon_length}
        for v in ds_to_reshape.data_vars:

            these_dims = dims.copy()
            if "level" not in ds_to_reshape[v].dims:
                these_dims.pop("level")
            reshaped_var = ds_to_reshape[v].values.reshape(tuple(these_dims.values()))
            data_vars[v] = (list(these_dims.keys()), reshaped_var)

        reshaped = xr.Dataset(
            data_vars=data_vars, coords={"time": ds_to_reshape["time"].values}
        )
        reshaped["latitude"] = (("y", "x"), lats)
        reshaped["longitude"] = (("y", "x"), lons)
        reshaped = reshaped.set_coords(["latitude", "longitude"])

    else:
        lats = ds_to_reshape.latitude.values
        lons = ds_to_reshape.longitude.values
        sort_index = np.lexsort((lons, lats))
        ds_to_reshape = ds_to_reshape.isel(cell=sort_index)

        lat_length = len(np.unique(ds_to_reshape.latitude.values))
        lon_length = len(np.unique(ds_to_reshape.longitude.values))
        time_length = len(ds["time"].values)

        lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
        lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))
        lat_1d = lats[:, 0]
        lon_1d = lons[0, :]

        data_vars = {}
        dims = {"time": time_length, "level": len(ds_to_reshape["level"]), "latitude": lat_length, "longitude": lon_length}
        for v in ds_to_reshape.data_vars:

            these_dims = dims.copy()
            if "level" not in ds_to_reshape[v].dims:
                these_dims.pop("level")
            reshaped_var = ds_to_reshape[v].values.reshape(tuple(these_dims.values()))
            data_vars[v] = (list(these_dims.keys()), reshaped_var)

        reshaped = xr.Dataset(
            data_vars=data_vars, coords={"latitude": lat_1d, "longitude": lon_1d}
        )

    return make_contiguous(reshaped)


def make_contiguous(
    reshaped,
):
    """
    xesmf was complaining about array not being in C format?
    apparently just a performance issue - but was tired of getting the warnings :)
    """
    for var in reshaped.data_vars:
        reshaped[var].data = np.ascontiguousarray(reshaped[var].values)
    return reshaped


def final_steps(ds: xr.Dataset, time: xr.DataArray) -> xr.Dataset:
    """
    Add helpful attributes and reorganize dimensions for verification pipelines.

    Args:
        ds (xr.Dataset): Input dataset.
        time (xr.DataArray): Time coordinate.

    Returns:
        xr.Dataset: Dataset with necessary attributes for verification pipelines.
    """
    ds.attrs["forecast_reference_time"] = str(ds["time"][0].values)
    if {"x", "y"}.issubset(ds.dims):
        return ds.transpose("time", "level", "y", "x")
    elif {"latitude", "longitude"}.issubset(ds.dims):
        return ds.transpose("time", "level", "latitude", "longitude")


def regrid_ds(ds_to_regrid: xr.Dataset, ds_out: xr.Dataset, weights_filename: str | None = None) -> xr.Dataset:
    """
    Regrid a dataset.

    Args:
        ds_to_regrid (xr.Dataset): Input dataset to regrid.
        ds_out (xr.Dataset): Target grid.

    Returns:
        xr.Dataset: Regridded dataset.
    """
    rename = {"latitude": "lat", "longitude": "lon"}
    dorename = False
    for key, val in rename.items():
        if key in ds_to_regrid:
            dorename = True
            ds_to_regrid = ds_to_regrid.rename({key: val})
        if key in ds_out:
            ds_out = ds_out.rename({key: val})


    ds_to_regrid = get_bounds(ds_to_regrid)
    ds_out = get_bounds(ds_out)

    rename_bounds = {"x_vertices": "x_b", "y_vertices": "y_b"}
    for key, val in rename_bounds.items():
        if key in ds_to_regrid:
            ds_to_regrid = ds_to_regrid.rename({key: val})

    ds_to_regrid = make_contiguous(ds_to_regrid)

    print(f"ds_in\n{ds_to_regrid}\n\n ---->\nds_out\n{ds_out}\n")

    reuse_weights = False
    if weights_filename is not None:
        if os.path.isfile(weights_filename):
            reuse_weights = True

    regridder = xe.Regridder(
        ds_to_regrid,
        ds_out,
        method="conservative",
        unmapped_to_nan=True,  # this makes sure anything out of conus is nan instead of zero when regridding conus only
        filename=weights_filename,
        reuse_weights=reuse_weights,
    )
    result = regridder(ds_to_regrid)
    if dorename:
        for val, key in rename.items():
            result = result.rename({key: val})
    return result


def get_conus_ds_out(
    global_ds: xr.Dataset,
    conus_ds: xr.Dataset,
    global_info: dict,
) -> xr.Dataset:
    """
    Create conus dataset on global grid.
    This will then be used for regridding high-res conus to global res.
    That will then be inserted into global domain so it's all the same resolution for verification.

    Args:
        global_ds (xr.Dataset): Global dataset.
        conus_ds (xr.Dataset): CONUS dataset.
        global_info (dict): Necessary information for global grid.

    Returns:
        xr.Dataset: Output dataset with CONUS grid.
    """
    res = global_info["res"]
    lat_min = global_info["lat_min"]
    lon_min = global_info["lon_min"]
    lat_max = global_info["lat_max"]
    lon_max = global_info["lon_max"]

    return xr.Dataset(
        {
            "latitude": (
                ["latitude"],
                np.arange(lat_min, lat_max, res),
                {"units": "degrees_north"},
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_min, lon_max, res),
                {"units": "degrees_east"},
            ),
        }
    )


def flatten_grid(ds_to_flatten: xr.Dataset) -> xr.Dataset:
    """
    Flatten a 2D lat-lon gridded dataset back to a 1D 'values' coordinate.
    This is necessary to eventually combine global and conus back together
        after high-res conus has been regridded to global res.

    Args:
        ds_to_flatten (xr.Dataset): Dataset with 2D lat/lon grid.

    Returns:
        xr.Dataset: Flattened dataset with 'values' dimension.
    """
    logger.info(f"ds_to_flatten\n{ds_to_flatten}")
    ds = ds_to_flatten.stack(cell2d=("latitude", "longitude"))
    ds = ds.dropna(dim="cell2d", how="any")

    ds["cell"] = xr.DataArray(
        np.arange(len(ds["cell2d"])),
        coords=ds["cell2d"].coords,
        dims=ds["cell2d"].dims,
        attrs={
            "description": f"logical index for 'cell2d', which is a flattened lon x lat array",
        },
    )
    ds = ds.swap_dims({"cell2d": "cell"})

    # For some reason, there's a failure when trying to store this multi-index
    # it's not needed in Anemoi, so no need to keep it anyway.
    ds = ds.drop_vars("cell2d")
    logger.info(f"after:\n{ds}")
    return ds


def combine_lam_w_global(
    ds_nested_w_lam_cutout: xr.Dataset, ds_lam_w_global_res: xr.Dataset
) -> xr.Dataset:
    """
    Combine LAM (regridded to global res) and global regions into a single dataset.

    Args:
        ds_nested_w_lam_cutout (xr.Dataset): Global portion of dataset.
        ds_lam_w_global_res (xr.Dataset): Regridded LAM portion of dataset.

    Returns:
        xr.Dataset: Combined dataset.
    """
    cell = len(ds_lam_w_global_res.cell) + np.arange(len(ds_nested_w_lam_cutout.cell))
    ds_nested_w_lam_cutout["cell"] = xr.DataArray(
        cell,
        coords={"cell": cell},
    )
    ds_nested_w_lam_cutout.to_netcdf("global_cutout.nc")
    ds_lam_w_global_res.to_netcdf("lam_lowres.nc")
    logger.info(f"cutout\n{ds_nested_w_lam_cutout}")
    logger.info(f"lam\n{ds_lam_w_global_res}")
    return xr.concat([ds_nested_w_lam_cutout, ds_lam_w_global_res], dim="cell")


def postprocess_lam_only(
    ds_nested: xr.Dataset,
    lam_index: int,
    levels: List[int],
    lcc_info: bool,
) -> xr.Dataset:
    """
    Postprocess LAM-only data.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        levels (List[int]): List of levels to process.
        lcc_info (bool): Flag if lcc_flag grid (e.g. HRRR).

    Returns:
        xr.Dataset: Processed LAM dataset ready for verification :)
    """
    time = ds_nested["time"]

    ds_lam = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    ds_lam = create_2D_grid(
        ds=ds_lam, lcc_info=lcc_info
    )
    logger.info(f"after 2D grid: \n {ds_lam}\n")
    ds_lam = final_steps(ds=ds_lam, time=time)
    return ds_lam


def postprocess_global(
    ds_nested: xr.Dataset,
    lam_index: int,
    lcc_info: dict,
    global_info: dict,
    weights_filename: str = "conservative_weights.nc",
) -> xr.Dataset:
    """
    Postprocess global data.
    This will output a global ds, and the LAM region has been regridded to global res within it.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        lcc_info (dict): Necessary information for a LCC grid.
        lcc_info (dict): Necessary information for the global grid.

    Returns:
        xr.Dataset: Post-processed global dataset.
    """
    time = ds_nested["time"]

    # create lam only ds and global only ds (lam has been cut out)
    lam_ds = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    global_ds = mask_values(
        area_to_return="global", ds_nested=ds_nested, lam_index=lam_index
    )

    ## take lam from 1D to 2D (values dim -> lat/lon or x/y dims)
    #lam_ds = create_2D_grid(
    #    ds=lam_ds, lcc_info=lcc_info
    #)

    ## create blank grid over conus that matches global resolution
    #ds_out_conus = get_conus_ds_out(global_ds, lam_ds, global_info=global_info)

    # regrid lam to match global resolution
    lam_ds_regridded = regrid_ds(ds_to_regrid=lam_ds, ds_out=global_ds.coords.to_dataset(), weights_filename=weights_filename)
    global_ds.to_netcdf("global.nc")
    lam_ds_regridded.to_netcdf("lam.global.nc")

    # flatten regridded lam back to 1D (lat/lon dims -> values dim)
    # necessary to concat it back to global grid
    ds_lam_regridded_flattened = flatten_grid(
        ds_to_flatten=lam_ds_regridded,
    )

    # combine global ds and regridded lam ds together
    ds_combined = combine_lam_w_global(
        ds_nested_w_lam_cutout=global_ds, ds_lam_w_global_res=ds_lam_regridded_flattened
    )

    # go back to 2D again (lots of gynmastics here!!)
    ds_combined = create_2D_grid(ds=ds_combined)

    ds_combined = final_steps(ds=ds_combined, time=time)
    ds_combined.to_netcdf("combined.nc")

    return ds_combined


def run(
    initialization: pd.Timestamp,
    config,
):
    """
    Run full pipeline.

    """
    vars_of_interest = config["vars_of_interest"]
    levels = config["levels"]
    lam_index = config["lam_index"]
    lcc_info = config["lcc_info"]
    global_info = config["global_info"]
    raw_inference_files_base_path = config["raw_inference_files_base_path"]

    file_date = datetime.fromisoformat(initialization).strftime("%Y-%m-%dT%H")
    file_name = f"{file_date}.240h"

    ds_nested = open_raw_inference(
        path_to_raw_inference=f"{raw_inference_files_base_path}/{file_name}.nc"
    )
    ds_nested = ds_nested[vars_of_interest]

    lam_ds = postprocess_lam_only(
        ds_nested=ds_nested,
        lam_index=lam_index,
        levels=levels,
        lcc_info=lcc_info,
    )

    lam_ds.to_netcdf(f"lam_{file_name}.nc")

    global_ds = postprocess_global(
        ds_nested=ds_nested,
        lam_index=lam_index,
        levels=levels,
        lcc_info=lcc_info,
        global_info=global_info,
    )

    global_ds.to_netcdf(f"global_{file_name}.nc")

    # TODO - revisit if this is how we want to be saving files out?

    return


def run_postprocessing(config_filename=None):
    if len(sys.argv) != 2 and config_filename is None:
        raise Exception("Did not get an argument. Usage is:\npython run_postprocessing.py config.yaml")

    config_filename = sys.argv[1] if config_filename is None else config_filename
    setup_simple_log()
    config = open_yaml_config(config_filename)

    dates = pd.date_range(start=config["start_date"], end=config["end_date"], freq=config["freq"])

    logger.info(f"Postprocessing inference with initial condition dates:\n{dates}")

    for i in dates:
        run(
            initialization=str(i),
            config=config,
        )
