import os

import numpy as np
import xarray as xr
import xesmf

from ufs2arco.transforms.horizontal_regrid import get_bounds


def get_xy():
    xds = xr.open_zarr("/pscratch/sd/t/timothys/nested-eagle/v0/data/hrrr.zarr")
    return {"x": xds["x"].isel(variable=0,drop=True).load(), "y": xds["y"].isel(variable=0,drop=True).load()}

def trim_xarray_edge(xds, trim_edge):
    assert all(key in xds for key in ("x", "y"))
    xds["x"].load()
    xds["y"].load()
    condx = ( (xds["x"] > trim_edge[0]-1) & (xds["x"] < xds["x"].max().values-trim_edge[1]+1) ).compute()
    condy = ( (xds["y"] > trim_edge[2]-1) & (xds["y"] < xds["y"].max().values-trim_edge[3]+1) ).compute()
    xds = xds.where(condx & condy, drop=True)
    return xds

def _expand_dataset(xds, eds, dims):

    for varname in xds.data_vars:
        these_dims = dims.copy()
        if "level" not in xds[varname].dims:
            these_dims.pop("level")
        eds[varname] = xr.DataArray(
            xds[varname].values.reshape(tuple(these_dims.values())),
            dims=tuple(these_dims.keys()),
            attrs=xds[varname].attrs.copy(),
        )
    return eds

def expand_global(xds):
    n_lat = len(np.unique(xds["lat"]))
    n_lon = len(np.unique(xds["lon"]))

    gds = xr.Dataset()
    gds["time"] = xds["time"]
    gds["level"] = xds["level"]
    gds["lat"] = np.unique(xds["lat"].values)
    gds["lon"] = np.unique(xds["lon"].values)

    dims = {
        "time": len(xds["time"]),
        "level": len(xds["level"]),
        "lat": n_lat,
        "lon": n_lon,
    }
    gds = _expand_dataset(xds, gds, dims)
    return gds


def expand_lam(xds):

    xy = get_xy()
    lds = xr.Dataset()
    lds["x"] = np.unique(xy["x"])
    lds["y"] = np.unique(xy["y"])
    lds["time"] = xds["time"]
    lds["level"] = xds["level"]
    lds = trim_xarray_edge(lds, trim_edge=(10,11,10,11))
    n_x = len(lds["x"])
    n_y = len(lds["y"])
    for cname in ["lat", "lon"]:
        lds[cname] = xr.DataArray(
            xds[cname].values.reshape( (n_y, n_x) ),
            dims=("y", "x"),
        )
        lds = lds.set_coords(cname)
    dims = {
        "time": len(xds["time"]),
        "level": len(xds["level"]),
        "y": n_y,
        "x": n_x,
    }
    lds = _expand_dataset(xds, lds, dims)
    return lds


def regrid_nested_to_global(
    nds: xr.Dataset,
    ds_out: xr.Dataset,
    lam_index: int,
    regrid_weights_filename: str | None = None,
) -> xr.Dataset:

    # 1. Rename to lon/lat for xesmf
    rename = {"latitude": "lat", "longitude": "lon"}
    for key, val in rename.items():
        if key in nds:
            nds = nds.rename({key: val})
        if key in ds_out:
            ds_out = ds_out.rename({key: val})

    # 2. Split LAM / global and put on 2D grid
    lds = expand_lam(nds.sel(cell=slice(lam_index)))
    ds_out = expand_global(ds_out)

    lds = get_bounds(lds)
    ds_out = get_bounds(ds_out)

    rename_bounds = {"x_vertices": "x_b", "y_vertices": "y_b"}
    for key, val in rename_bounds.items():
        if key in lds:
            lds = lds.rename({key: val})

    #lds = make_contiguous(lds)

    print(f"ds_in\n{lds}\n\n ---->\nds_out\n{ds_out}\n")

    reuse_weights = False
    if regrid_weights_filename is not None:
        if os.path.isfile(regrid_weights_filename):
            reuse_weights = True

    regridder = xesmf.Regridder(
        lds,
        ds_out,
        method="conservative",
        unmapped_to_nan=True,  # this makes sure anything out of conus is nan instead of zero when regridding conus only
        filename=regrid_weights_filename,
        reuse_weights=reuse_weights,
    )
    result = regridder(lds)

    # Rename back to latitude/longitude for the rest of the code
    for val, key in rename.items():
        result = result.rename({key: val})

    print(f"result!!!\n{result}")
    result.to_netcdf("lam.global.nc")
    nds.sel(cell=slice(lam_index, None)).to_netcdf("global.nc")
    raise
    return result
