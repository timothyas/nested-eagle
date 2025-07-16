"""TODO List for this

* separate comparisons to ERA5/Replay and GFS/HRRR... this is getting nasty
* pass the model name as argument to main
* variable list should be an input argument
* eventually we'll want to pass the dot size as an argument too, TBD
"""
import os
import sys
import logging

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cmocean

try:
    import xmovie
except ImportError:
    print("Could not import xmovie, can't use mode='movie'")

from anemoi.datasets import open_dataset as open_anemoi_dataset
from ufs2arco.utils import expand_anemoi_to_dataset, convert_anemoi_inference
from eagle.log import setup_simple_log

_projection = ccrs.Orthographic(
    central_longitude = -120,
    central_latitude = 20,
)

def get_extend(xds, vmin=None, vmax=None):
    minval = []
    maxval = []
    for key in xds.data_vars:
        minval.append(xds[key].min().values)
        maxval.append(xds[key].max().values)
    minval = np.min(minval)
    maxval = np.max(maxval)
    vmin = minval if vmin is None else vmin
    vmax = maxval if vmax is None else vmax

    extend = "neither"
    if minval < vmin:
        extend = "min"
    if maxval > vmax:
        extend = "max" if extend == "neither" else "both"
    return extend, vmin, vmax

def get_precip_kwargs():
    n = 1
    levels = np.concatenate(
        [
            np.linspace(0, .1, 2*n),
            np.linspace(.1, 1, 5*n),
            np.linspace(1, 10, 5*n),
            np.linspace(10, 50, 3*n),
            #np.linspace(50, 80, 2),
        ],
    )
    norm = BoundaryNorm(levels, len(levels)+1)
    cmap = plt.get_cmap("cmo.rain", len(levels)+1)
    return {"norm": norm, "cmap": cmap, "cbar_kwargs": {"ticks": [0, 1, 10, 50]}}

def draw_box(ax, hds, trim_lam_edge=None, **kwargs):

    assert hds.stack_order == ["y", "x"]
    lons = hds["longitudes"].values.reshape(hds.field_shape)
    lats = hds["latitudes"].values.reshape(hds.field_shape)

    if trim_lam_edge is not None:
        lons = lons[trim_lam_edge[0]:-trim_lam_edge[1], trim_lam_edge[2]:-trim_lam_edge[3]]
        lats = lats[trim_lam_edge[0]:-trim_lam_edge[1], trim_lam_edge[2]:-trim_lam_edge[3]]

    n_x = lons.shape[1]
    n_y = lons.shape[0]

    xidx = np.arange(0, n_x, 10)
    if xidx[-1] != n_x - 1:
        xidx = np.append(xidx, n_x - 1)
    yidx = np.arange(0, n_y, 10)
    if yidx[-1] != n_y - 1:
        yidx = np.append(yidx, n_y - 1)

    kw = {
        "color": "gray",
        "transform": ccrs.PlateCarree(),
        "lw": 2,
        "alpha": .8,
    }
    kw.update(kwargs)

    for idx in [0, -1]:
        ax.plot(lons[yidx, idx], lats[yidx, idx], **kw)
        ax.plot(lons[idx, xidx], lats[idx, xidx], **kw)


def nested_scatter(ax, xds, varname, n_conus, **kwargs):
    mappables = []
    for slc, s in zip(
        [slice(None, n_conus), slice(n_conus, None)],
        [1/4, 12],
    ):

        p = ax.scatter(
            xds.longitudes.isel(cell=slc),
            xds.latitudes.isel(cell=slc),
            c=xds[varname].isel(cell=slc),
            s=s,
            transform=ccrs.PlateCarree(),
            **kwargs
        )
        mappables.append(p)
    return mappables

def plot_single_timestamp(xds, fig, time, *args, **kwargs):

    axs = []

    vtime = xds["time"].isel(time=time).values
    stime = str(vtime)[:13]

    # get these extra options
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    extend = kwargs.pop("extend", None)
    t0 = kwargs.pop("t0", "")
    hds = kwargs.pop("hds", None)
    trim_lam_edge = kwargs.pop("trim_lam_edge", None)

    # Create axes
    ax = fig.add_subplot(1, 2, 1, projection=_projection)

    # Plot Truth
    # Note that this scatter is very slow for movies
    # But... it is the truest comparison to the other plot
    # We could move to datashader eventually
    if xds.truth.nice_name in ["ERA5", "Replay"]:
        p = ax.scatter(
            truth["x"],
            truth["y"],
            c=xds.truth.isel(time=time),
            s=.5,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
    else:
        p = nested_scatter(ax, xds.isel(time=time), "truth", n_conus=xds.truth.attrs["n_conus"], **kwargs)
        draw_box(ax, hds, trim_lam_edge=trim_lam_edge)

    ax.set(title=xds.truth.nice_name)
    axs.append(ax)

    # Plot model
    ax = fig.add_subplot(1, 2, 2, projection=_projection)

    pp = nested_scatter(ax, xds.isel(time=time), "prediction", n_conus=xds.truth.attrs["n_conus"], **kwargs)
    draw_box(ax, hds, trim_lam_edge=trim_lam_edge)
    ax.set(title=xds.prediction.nice_name)
    axs.append(ax)

    # now the colorbar
    [ax.set(xlabel="", ylabel="") for ax in axs]
    [ax.coastlines("50m") for ax in axs]

    label = xds.attrs.get("label", "")
    label += f"\nt0: {t0}"
    label += f"\nvalid: {stime}"
    fig.colorbar(
        pp[0],
        ax=axs,
        orientation="horizontal",
        shrink=.8,
        pad=0.05,
        aspect=35,
        label=label,
        extend=extend,
        **cbar_kwargs,
    )
    fig.set_constrained_layout(True)

    return None, None

def calc_wind_speed(xds):
    if "ugrd10m" in xds:
        ws = np.sqrt(xds["ugrd10m"]**2 + xds["vgrd10m"]**2)
    elif "u10" in xds:
        ws = np.sqrt(xds["u10"]**2 + xds["v10"]**2)
    else:
        ws = np.sqrt(xds["10m_u_component_of_wind"]**2 + xds["10m_v_component_of_wind"]**2)
    ws.attrs["units"] = "m/s"
    ws.attrs["long_name"] = "10m Wind Speed"
    xds["10m_wind_speed"] = ws

    if "80m_u_component_of_wind" in xds:
        ws80 = np.sqrt(xds["80m_u_component_of_wind"]**2 + xds["80m_v_component_of_wind"]**2)
        xds["80m_wind_speed"] = ws80
        xds["80m_wind_speed"].attrs["units"] = "m/s"
    return xds

def get_reanalysis(name):
    if name.lower() == "era5":
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        rename = {}
    elif name.lower() == "replay":
        url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"
        rename = {"pfull": "level", "grid_yt": "latitude", "grid_xt": "longitude"}

    truth = xr.open_zarr(
        url,
        storage_options={"token": "anon"},
    )
    truth = truth.rename(rename)
    truth.attrs["name"] = name
    truth["x"], truth["y"] = np.meshgrid(truth["longitude"], truth["latitude"])
    return truth

def get_truth(name, t0, tf, trim_lam_edge=None):
    if name in ["era5", "replay"]:
        truth = get_reanalysis(name)
    else:
        data_dir = "/pscratch/sd/t/timothys/nested-eagle/v0/data"
        kw = {}
        if trim_lam_edge is not None:
            kw = {"trim_edge": trim_lam_edge}
        ads = open_anemoi_dataset(
            cutout=[
                {
                    "join":
                    [
                        f"{data_dir}/hrrr.analysis.zarr",
                        f"{data_dir}/hrrr.forecast.zarr",
                    ],
                    **kw,
                },
                {
                    "join":
                    [
                        f"{data_dir}/gfs.analysis.zarr",
                        f"{data_dir}/gfs.forecast.zarr",
                    ],
                },
            ],
            adjust="all",
            min_distance_km=0,
        )

        start = ads.to_index(date=t0, variable=0)[0]
        end = ads.to_index(date=tf, variable=0)[0] + 1
        truth = xr.DataArray(
            ads[start:end,:,:,:],
            coords={
                "time": np.arange(end-start),
                "variable": np.arange(ads.shape[1]),
                "ensemble": np.arange(ads.shape[2]),
                "cell": np.arange(ads.shape[3]),
            },
            dims=("time", "variable", "ensemble", "cell"),
        ).load().squeeze()
        truth = expand_anemoi_to_dataset(truth, ads.variables)
        truth["dates"] = xr.DataArray(
            ads.dates[start:end],
            dims="time",
        )
        truth = truth.swap_dims({"time": "dates"}).drop_vars("time").rename({"dates": "time"})

        for varname, unit in zip(
            ["t2m", "sh2", "accum_tp", "u10", "v10", "u", "v", "w", "u80", "v80"],
            ["K", "kg/kg", "kg/m$^2$", "m/s", "m/s", "m/s", "m/s", "m/s", "m/s", "m/s"],
        ):
            truth[varname].attrs["units"] = unit
        truth.attrs["n_conus"] = ads.grids[0]

    truth = rename_short_to_long(truth)
    truth = calc_wind_speed(truth)
    return truth



def rename_short_to_long(xds):

    rename = {
        "accum_tp": "total_precipitation_6hr",
        "t2m": "2m_temperature",
        "sh2": "2m_specific_humidity",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "u80": "80m_u_component_of_wind",
        "v80": "80m_v_component_of_wind",
    }
    return xds.rename({key: val for key, val in rename.items() if key in xds})

def main(
    read_path,
    store_dir,
    t0,
    tf,
    ifreq=1,
    mode="figure", # or movie
    trim_lam_edge=None,
):
    """A note about t0
    In the inference yaml, I think this means "the very first initial condition"... makes sense

    But when I'm visualizing the data, I think about t0 as the last initial condition...
    as in the last data given to the model before making a forecast.
    So... the t0 given here is that one... the last IC.
    """

    setup_simple_log()

    assert mode in ["figure", "movie"]

    plot_options = {
        "total_precipitation_6hr": get_precip_kwargs(),
        "80m_wind_speed": {
            "cmap": "cmo.tempo_r",
            "vmin": 0,
            "vmax": 20,
        },
        "2m_temperature": {
            "cmap": "cmo.thermal",
            "vmin": -10,
            "vmax": 30,
        },
        "10m_wind_speed": {
            "cmap": "cmo.tempo_r",
            "vmin": 0,
            "vmax": 20,
        },
        "2m_specific_humidity": {
            "cmap": "cmo.rain",
            "vmin": 0,
            "vmax": 0.025,
        },
    }

    logging.info(f"Time Bounds:\n\tt0 = {t0}\n\ttf = {tf}\n")
    psl = xr.open_dataset(read_path)
    psl = psl.sel(time=slice(t0, tf))
    psl = psl.isel(time=slice(None, None, ifreq))
    psl = psl.rename({"longitude": "longitudes", "latitude": "latitudes"})
    psl = psl.set_coords(["longitudes", "latitudes"])
    psl = psl.rename({"values": "cell"})
    psl = rename_short_to_long(psl)
    psl = calc_wind_speed(psl)
    psl.attrs["nice_name"] = "Prediction: Nested-EAGLE-v0.30k"

    hrrr = xr.open_zarr("/pscratch/sd/t/timothys/nested-eagle/v0/data/hrrr.analysis.zarr")

    logging.info(f"Ready to make {mode}s with dataset:\n{psl}\n")

    for tname in ["GFS/HRRR"]: #["ERA5"]:

        truth = get_truth(tname, t0=t0, tf=tf, trim_lam_edge=trim_lam_edge)

        logging.info(f"Retrieved truth = {tname}\n{truth}\n")
        fig_dir = os.path.join(store_dir, f"{mode}s", f"{tname.lower().replace('/','-')}-vs-nested")
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
            logging.info(f"Created fig_dir: {fig_dir}")

        for varname, options in plot_options.items():

            logging.info(f"Plotting {varname} with options")
            for key, val in options.items():
                logging.info(f"\t{key}: {val}")

            ds = xr.Dataset({
                "prediction": psl[varname].load(),
                "truth": truth[varname].sel(
                    time=psl.time.values,
                ).load(),
            })
            ds["prediction"].attrs["nice_name"] = psl.nice_name
            ds["truth"].attrs["nice_name"] = tname
            ds["truth"].attrs["n_conus"] = truth.attrs["n_conus"]

            # Convert to degC
            if varname[:3] == "tmp" or "temperature" in varname:
                for key in ds.data_vars:
                    ds[key] -= 273.15
                    ds[key].attrs["units"] = "degC"

                logging.info(f"\tconverted {varname} K -> degC")

            # Convert to mm->m
            if "total_precipitation" in varname and tname == "ERA5":
                ds["truth"] *= 1000
                ds["truth"].attrs["units"] = "m"

                logging.info(f"\tconverted ERA5 {varname} mm -> m")

            label = " ".join([x.capitalize() for x in varname.split("_")])
            ds.attrs["label"] = f"{label} ({ds.truth.units})"

            # colorbar extension options
            options["extend"], vmin, vmax = get_extend(
                ds,
                vmin=options.get("vmin", None),
                vmax=options.get("vmax", None),
            )
            logging.info(f"\tcolorbar extend = {options['extend']}")

            # precip is weird, since we don't do vmin/vmax, we do BoundaryNorm colorbar map blah blah
            # since we know it's bounded to be positive in anemoi... at least in this model..
            # then just worry about max
            if "total_precipitation" in varname:
                options["extend"] = "max" if vmax > 50 else "neither"
                logging.info(f"\ttotal_precipitation hack: setting extend based on upper limit of 50")

            options["t0"] = t0
            options["hds"] = hrrr
            options["trim_lam_edge"] = trim_lam_edge

            dpi = 300
            width = 10
            height = 7
            pixelwidth = width*dpi
            pixelheight = height*dpi

            if mode == "figure":

                fig = plt.figure(figsize=(width, height))
                itime = list(pd.Timestamp(x) for x in ds["time"].values).index(pd.Timestamp(tf))
                plot_single_timestamp(
                    xds=ds,
                    fig=fig,
                    time=itime,
                    **options,
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.jpeg"
                fig.savefig(fname, dpi=dpi, bbox_inches="tight")
                logging.info(f"Stored figure at: {fname}\n")

            else:
                mov = xmovie.Movie(
                    ds,
                    plot_single_timestamp,
                    framedim="time",
                    input_check=False,
                    pixelwidth=pixelwidth,
                    pixelheight=pixelheight,
                    dpi=dpi,
                    **options
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.gif"
                mov.save(
                    fname,
                    progress=True,
                    overwrite_existing=True,
                    remove_frames=True,
                    framerate=10,
                    gif_framerate=10,
                    remove_movie=False,
                    gif_palette=True,
                    gif_scale=["trunc(iw/2)", "trunc(ih/2)"],
                )
                logging.info(f"Stored movie at: {fname}\n")
