import logging
import sys

import numpy as np
from scipy.spatial import SphericalVoronoi

import xarray as xr
import pandas as pd

import ufs2arco.utils

from .log import setup_simple_log
from .utils import open_yaml_config

logger = logging.getLogger("eagle")


def get_xy():
    xds = xr.open_zarr("/pscratch/sd/t/timothys/nested-eagle/v0/data/hrrr.analysis.zarr")
    return {"x": xds["x"].load(), "y": xds["y"].load()}

_extra_coords = get_xy()

def drop_forcing_vars(xds):
    for key in [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "orog",
        "orography",
        "geopotential_at_surface",
        "land_sea_mask",
        "lsm",
        "insolation",
        "cos_solar_zenith_angle",
    ]:
        if key in xds:
            xds = xds.drop_vars(key)
    return xds


def get_gridcell_area_weights(xds, unit_mean=True, radius=1, center=np.array([0,0,0]), threshold=1e-12):
    """This is a nice code block copied from anemoi-graphs"""

    x = radius * np.cos(np.deg2rad(xds["latitude"])) * np.cos(np.deg2rad(xds["longitude"]))
    y = radius * np.cos(np.deg2rad(xds["latitude"])) * np.sin(np.deg2rad(xds["longitude"]))
    z = radius * np.sin(np.deg2rad(xds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    area_weight = sv.calculate_areas()
    if unit_mean:
        area_weight /= area_weight.mean()

    return area_weight


def subsample(xds, levels=None, vars_of_interest=None):
    """Subsample vertical levels and variables
    """

    if levels is not None:
        xds = xds.sel(level=levels)

    if vars_of_interest is not None:
        xds = xds[vars_of_interest]
    else:
        xds = drop_forcing_vars(xds)

    return xds


def trim_xarray_edge(xds, trim_edge):
    assert all(key in xds for key in ("x", "y"))
    xds["x"].load()
    xds["y"].load()
    condx = ( (xds["x"] > trim_edge[0]-1) & (xds["x"] < xds["x"].max().values-trim_edge[1]+1) ).compute()
    condy = ( (xds["y"] > trim_edge[2]-1) & (xds["y"] < xds["y"].max().values-trim_edge[3]+1) ).compute()
    xds = xds.where(condx & condy, drop=True)
    return xds



def open_anemoi_dataset(path, trim_edge=None, levels=None, vars_of_interest=None):

    xds = xr.open_zarr(path)
    vds = ufs2arco.utils.expand_anemoi_dataset(xds, "data", xds.attrs["variables"])
    for key in ["x", "y"]:
        if key in xds:
            vds[key] = xds[key] if "variable" not in xds[key].dims else xds[key].isel(variable=0, drop=True)
            vds = vds.set_coords(key)

    vds = subsample(vds, levels, vars_of_interest)
    if trim_edge is not None:
        vds = trim_xarray_edge(vds, trim_edge)
    return vds

def open_anemoi_inference_dataset(path, model_type, lam_index=None, levels=None, vars_of_interest=None, trim_edge=None):
    assert model_type in ("nested-lam", "nested-global", "global")

    ids = xr.open_dataset(path, chunks="auto")
    xds = ufs2arco.utils.convert_anemoi_inference_dataset(ids)

    if "nested" in model_type:
        assert lam_index is not None
        if "lam" in model_type:
            xds = xds.isel(cell=slice(lam_index))

        else:
            xds = xds.isel(cell=slice(lam_index,None))
            raise NotImplementedError("Need to put in the cutout/regridding stuff for nested-global")

    xds = subsample(xds, levels, vars_of_interest)
    xds = xds.load()
    if trim_edge is not None:
        for key in ["x", "y"]:
            if key in ids:
                xds[key] = ids[key] if "variable" not in ids[key].dims else ids[key].isel(variable=0, drop=True)
                xds = xds.set_coords(key)
            else:
                xds[key] = _extra_coords[key]
                xds = xds.set_coords(key)
        xds = trim_xarray_edge(xds, trim_edge)
    return xds

def open_forecast_zarr_dataset(path, t0, levels=None, vars_of_interest=None, trim_edge=None):
    """This is for non-anemoi forecast datasets, for example HRRR forecast data preprocessed by ufs2arco"""

    xds = xr.open_zarr(path, decode_timedelta=True)
    xds = xds.sel(t0=t0).squeeze(drop=True)
    xds["time"] = xr.DataArray(
        [pd.Timestamp(t0) + pd.Timedelta(hours=fhr) for fhr in xds.fhr.values],
        coords=xds.fhr.coords,
    )
    xds = xds.swap_dims({"fhr": "time"}).drop_vars("fhr")
    xds = subsample(xds, levels, vars_of_interest)

    # Comparing to anemoi, it's easier to flatten than unpack anemoi
    # this is
    if {"x", "y"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("y", "x"))
    elif {"longitude", "latitude"}.issubset(xds.dims):
        xds = xds.stack(cell2d=("latitude", "longitude"))
    else:
        raise KeyError("Unclear on the dimensions here")

    xds["cell"] = xr.DataArray(
        np.arange(len(xds.cell2d)),
        coords=xds.cell2d.coords,
    )
    xds = xds.swap_dims({"cell2d": "cell"})
    xds = xds.drop_vars(["cell2d", "t0", "valid_time"])
    xds = xds.load()
    if trim_edge is not None:
        xds = trim_xarray_edge(xds, trim_edge)
    return xds

def postprocess(xds, keep_t0=False):

    if "cell" not in xds.dims or keep_t0:
        t0 = pd.Timestamp(xds["time"][0].values)
        xds["t0"] = xr.DataArray(t0, coords={"t0": t0})
        xds = xds.set_coords("t0")
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds["fhr"] = xr.DataArray(
        xds["lead_time"].values.astype("timedelta64[h]").astype(int),
        coords=xds.time.coords,
        attrs={"description": "forecast hour, aka lead time in hours"},
    )
    xds = xds.swap_dims({"time": "fhr"}).drop_vars("time")
    xds = xds.set_coords("lead_time")
    return xds

def rmse(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(["cell", "ensemble"])
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds)

def spatial_rmse(target, prediction, weights=1., keep_t0=False):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean("ensemble")
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def mae(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(["cell", "ensemble"])
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds)


def spatial_mae(target, prediction, weights=1., keep_t0=False):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean("ensemble")
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    return postprocess(xds, keep_t0)


def compute_error_metrics():
    """Compute grid cell area weighted RMSE and MAE

    Note that the arguments documented here are passed via a config yaml as in

    Example:
        >>> python compute_error_metrics.py recipe.yaml

    Args:
        forecast_path (str): directory containing forecast datasets to compare against a verification dataset. For now, the convention is that, within this directory, each forecast is in a separate netcdf file named as "<initial_date>.<lead_time>.nc", where initial_date = "%Y-%m-%dT%H" and lead_time is defined below
        lead_time (str): a string indicating length, e.g. 240h or 90d, it doesn't matter what format, just make it the same as what was saved during forecast time
        verification_dataset_path (str): path to the zarr verification dataset
        model_type (str): "nested-lam", "nested-global", or "global"
        lam_index (int): number of points in nested domain that are dedicated to LAM
        output_path (str): directory to save rmse.nc and mae.nc
        start_date (str): date of first last IC to grab, in %Y-%m-%dTH format
        end_date (str): date of last last IC to grab, in %Y-%m-%dTH format
        freq (str): frequency over which to grab initial condition dates, passed to pandas.date_range
    """

    if len(sys.argv) != 2:
        raise Exception("Did not get an argument. Usage is:\npython compute_error_metrics.py recipe.yaml")

    config = open_yaml_config(sys.argv[1])
    setup_simple_log()

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    keep_spatial_t0 = config.get("keep_spatial_t0", False)
    subsample_kwargs = {
        "levels": config.get("levels", None),
        "vars_of_interest": config.get("vars_of_interest", None),
    }

    # Verification dataset
    vds = open_anemoi_dataset(
        path=config["verification_dataset_path"],
        trim_edge=config.get("trim_edge", None),
        **subsample_kwargs,
    )

    # Area weights
    if model_type == "global":
        latlon_weights = get_gridcell_area_weights(vds)

    elif model_type == "nested-lam":
        latlon_weights = 1. # Assume LAM is equal area

    elif model_type == "lam":
        latlon_weights = 1. # Assume LAM is equal area

    else:
        raise NotImplementedError

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    rmse_container = list()
    mae_container = list()
    spatial_rmse_container = list() if keep_spatial_t0 else None
    spatial_mae_container = list() if keep_spatial_t0 else None
    logger.info(f" --- Starting Metrics Computation --- ")
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        if config.get("from_anemoi", True):

            fds = open_anemoi_inference_dataset(
                f"{config['forecast_path']}/{st0}.{config['lead_time']}.nc",
                model_type=model_type,
                lam_index=lam_index,
                trim_edge=config.get("trim_forecast_edge", None),
                **subsample_kwargs,
            )
        else:

            fds = open_forecast_zarr_dataset(
                config["forecast_path"],
                t0=t0,
                trim_edge=config.get("trim_forecast_edge", None),
                **subsample_kwargs,
            )

        tds = vds.sel(time=fds.time.values).load()

        rmse_container.append(rmse(target=tds, prediction=fds, weights=latlon_weights))
        mae_container.append(mae(target=tds, prediction=fds, weights=latlon_weights))

        this_spatial_rmse = spatial_rmse(target=tds, prediction=fds, weights=latlon_weights, keep_t0=keep_spatial_t0)
        this_spatial_mae = spatial_mae(target=tds, prediction=fds, weights=latlon_weights, keep_t0=keep_spatial_t0)

        if spatial_rmse_container is None:
            spatial_rmse_container = this_spatial_rmse / len(dates)
            spatial_mae_container = this_spatial_mae / len(dates)

        else:
            if keep_spatial_t0:
                spatial_rmse_container.append(this_spatial_rmse)
                spatial_mae_container.append(this_spatial_mae)
            else:
                spatial_rmse_container += this_spatial_rmse / len(dates)
                spatial_mae_container += this_spatial_mae / len(dates)

        logger.info(f"\tDone with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")

    logger.info(f" --- Combining & Storing Results --- ")
    rmse_container = xr.concat(rmse_container, dim="t0")
    mae_container = xr.concat(mae_container, dim="t0")

    rmse_container.to_netcdf(f"{config['output_path']}/rmse.{config['model_type']}.nc")
    mae_container.to_netcdf(f"{config['output_path']}/mae.{config['model_type']}.nc")

    if keep_spatial_t0:
        spatial_rmse_container = xr.concat(spatial_rmse_container, dim="t0")
        fname = f"{config['output_path']}/spatial.rmse.perIC.{config['model_type']}.nc"
    else:
        fname = f"{config['output_path']}/spatial.rmse.{config['model_type']}.nc"

    spatial_rmse_container.to_netcdf(fname)

    if keep_spatial_t0:
        spatial_mae_container = xr.concat(spatial_mae_container, dim="t0")
        fname = f"{config['output_path']}/spatial.mae.perIC.{config['model_type']}.nc"
    else:
        fname = f"{config['output_path']}/spatial.mae.{config['model_type']}.nc"

    spatial_mae_container.to_netcdf(fname)

    logger.info(f" --- Done Storing Results at {config['output_path']} --- \n")
