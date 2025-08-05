import logging
import sys

import numpy as np
import xarray as xr

import pandas as pd

import ufs2arco.utils

from anemoi.training.diagnostics.plots import compute_spectra as compute_array_spectra, equirectangular_projection
from scipy.interpolate import griddata

from .log import setup_simple_log
from .metrics import open_anemoi_dataset, open_anemoi_inference_dataset, postprocess
from .utils import open_yaml_config

logger = logging.getLogger("eagle")


def compute_power_spectrum(xds, latlons, min_delta):

    pc_lat, pc_lon = equirectangular_projection(latlons)

    pc_lat = np.array(pc_lat)
    # Calculate delta_lat on the projected grid
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]
    min_delta_lat = np.min(abs(non_zero_delta_lat))

    if min_delta_lat < min_delta:
        min_delta_lat = min_delta

    # Define a regular grid for interpolation
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / min_delta_lat))
    n_pix_lon = (n_pix_lat - 1) * 2 + 1  # 2*lmax + 1
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    nds = dict()
    for varname in xds.data_vars:

        varlist = []
        for time in xds.time.values:
            yp = xds[varname].sel(time=time).values.squeeze()
            nan_flag = np.isnan(yp).any()

            method = "linear" if nan_flag else "cubic"
            yp_i = griddata((pc_lon, pc_lat), yp, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

            # Masking NaN values
            if nan_flag:
                mask = np.isnan(yp_i)
                if mask.any():
                    yp_i = np.where(mask, 0.0, yp_i)

            amplitude = np.array(compute_array_spectra(yp_i))
            varlist.append(amplitude)

        xamp = xr.DataArray(
            np.array(varlist),
            coords={"time": xds.time.values, "k": np.arange(len(amplitude))},
            dims=("time", "k",),
        )

        nds[varname] = xamp
    return postprocess(xr.Dataset(nds), keep_t0=False)


def compute_spectra():
    if len(sys.argv) != 2:
        raise Exception("Did not get an argument. Usage is:\npython compute_spectra.py recipe.yaml")

    config = open_yaml_config(sys.argv[1])
    setup_simple_log()

    # options used for verification and inference datasets
    model_type = config["model_type"]
    lam_index = config.get("lam_index", None)
    keep_spatial_t0 = config.get("keep_spatial_t0", False)
    min_delta = config.get("min_delta_lat", 0.0045)
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
    latlons = np.stack([vds["latitude"].values, vds["longitude"].values], axis=1)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])
    logger.info(f"Working with the following initial conditions\n{dates}\n")

    pspectra = None
    logger.info(f" --- Starting Spectra Computation --- ")
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

        this_pspectra = compute_power_spectrum(fds, latlons=latlons, min_delta=min_delta)

        if pspectra is None:
            pspectra = this_pspectra / len(dates)

        else:
            pspectra += this_pspectra / len(dates)

        logger.info(f"\tDone with {st0}")

    logger.info(f" --- Done Computing Spectra --- ")
    logger.info(f" --- Combining & Storing Results --- ")

    pspectra.to_netcdf(f"{config['output_path']}/spectra.predictions.{config['model_type']}.nc")
