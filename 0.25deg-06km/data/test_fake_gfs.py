import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from eagle.tools.data import open_anemoi_dataset

_global_dataset = "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/testing/gfs.zarr"
_fake_gfs_path =  "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/testing/gfs.with.hrrr.singlerez.zarr"

if __name__ == "__main__":

    date = "2023-02-10T06"

    gfs = open_anemoi_dataset(
        _global_dataset,
        model_type="global",
        t0=date,
        tf=date,
        vars_of_interest=["t2m", "u10", "v10"],
        reshape_cell_to_2d=True,
    )

    faker = open_anemoi_dataset(
        _fake_gfs_path,
        model_type="global",
        t0=date,
        tf=date,
        vars_of_interest=["t2m", "u10", "v10"],
        reshape_cell_to_2d=True,
    )

    diff = faker - gfs
    for key in gfs.data_vars:
        fig, axs = plt.subplots(3, 1, figsize=(6, 18))
        faker[key].plot(x="longitude", ax=axs[0])
        diff[key].plot(x="longitude", ax=axs[1])
        gfs[key].plot(x="longitude", ax=axs[2])
        fig.savefig(f"{key}.png")


    #tidx = reference.to_index(date=date, variable=0)[0]
    #print(reference.variables)
    #print(fds.attrs["variables"])
    #vidx = reference.variables.index("t2m")
    #faker = fds.data.values[0, vidx, 0, :]
    #gfs = reference[tidx, vidx, 0, :]
    #diff = faker - gfs
    #diff = diff.reshape([len(xds.latitude), len(xds.longitude)])
    #faker = faker.reshape([len(xds.latitude), len(xds.longitude)])
    #gfs = gfs.reshape([len(xds.latitude), len(xds.longitude)])
    #fig, axs = plt.subplots(3, 1, figsize=(6, 18))
    #axs[0].pcolormesh(faker)
    #axs[1].pcolormesh(diff)
    #axs[2].pcolormesh(gfs)
    #fig.savefig("diff.png")
