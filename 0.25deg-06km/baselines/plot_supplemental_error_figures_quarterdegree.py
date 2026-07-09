import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

from niceplots import plot_surface_vars, plot_level_vars, plot_selection, get_color


if __name__ == "__main__":

    plt.style.use("~/nice.mplstyle")

    fig_dir = "supplemental-figures/0.25deg"
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    metric = "rmse"
    valortest = "testing"

    kwargs = {
        "variables": [
            {"geopotential_height": 250}, {"wind_speed": 250}, {"temperature": 250}, {"specific_humidity": 250},
            {"geopotential_height": 500}, {"wind_speed": 500}, {"temperature": 500}, {"specific_humidity": 500},
            {"geopotential_height": 850}, {"wind_speed": 850}, {"temperature": 850}, {"specific_humidity": 850},
            "surface_pressure", "10m_wind_speed", "2m_temperature", "2m_specific_humidity",
        ],
        "nrows": 4,
    }

    for subregion in [
        "",
        ".conus",
        ".europe",
        ".northern_hemisphere",
        ".southern_hemisphere",
        ".tropics",
        ".polar_north",
        ".polar_south",
    ]:

        fileprefix = subregion.replace(".","") if subregion != "" else "global"

        figdict = {}
        figdict["Nested-EAGLE"] = xr.load_dataset(f"/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/production/gfs-hrrr/stage1c/inference-{valortest}/obs-metrics/{metric}.convobs.nested-global{subregion}.nc", decode_timedelta=True)
        figdict["ML-GFS-Base"] = xr.load_dataset(f"/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/production/gfs-only/stage1c/inference-{valortest}/obs-metrics/{metric}.convobs.global{subregion}.nc", decode_timedelta=True)
        figdict["GFS"] = xr.load_dataset(f"/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines/gfs-forecasts-vs-gfs-analysis/obs-metrics-{valortest}/{metric}.convobs.global{subregion}.nc", decode_timedelta=True)

        fig, axs = plot_selection(figdict, metric_name=metric.upper(), **kwargs)
        fig.savefig(f"{fig_dir}/{fileprefix}_{metric}.jpeg", dpi=300, bbox_inches="tight")
        fig.savefig(f"{fig_dir}/{fileprefix}_{metric}.pdf", bbox_inches="tight")
