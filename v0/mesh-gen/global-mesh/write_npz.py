import numpy as np
import xarray as xr

if __name__ == "__main__":

    gmesh = xr.load_dataset("latentx2.global1degree.nc")
    glon, glat = np.meshgrid(gmesh.lon, gmesh.lat)

    #gds = xr.Dataset()
    #gds["lon"] = xr.DataArray(
    #    glon.flatten(),
    #    coords={"lon": glon.flatten()},
    #)
    #gds["lat"] = xr.DataArray(
    #    glat.flatten(),
    #    coords={"lat": glat.flatten()},
    #)
    np.savez("latentx2.global1degree.npz", lon=glon.flatten(), lat=glat.flatten())
