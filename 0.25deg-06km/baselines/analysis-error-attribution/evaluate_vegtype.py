"""
Surface error by HRRR vegetation type (a categorical land/climate regime).

`vgtyp` (MODIS-IGBP class, integer) is sampled NEAREST to each station (it is
nominal -- no interpolation, no regression on the codes). For each class we
report the count-weighted mean log10(RMSE) for each model and the gap
(log-ratio gfs-only - gfs-hrrr) on the shared stations, with median elevation as
a terrain sanity check. A large gap concentrated in land-coupled vegetated
classes (vs water/barren) for 2m T but not 10m wind would fingerprint a better
HRRR land model.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import xesmf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate_error_gap as eg

SCRATCH = os.environ["SCRATCH"]
HRRR_ZARR = f"{SCRATCH}/nested-eagle/0.25deg-06km/data/native/hrrr.zarr"
HD = (f"{SCRATCH}/nested-eagle/0.25deg-06km/production/gfs-hrrr/stage1c"
      f"/inference-testing/spatial-obs-metrics")
MIN_PER_CLASS = 20
# MODIS-IGBP 20-category land-use names
NAMES = {1: "EvgNeedleForest", 2: "EvgBroadForest", 3: "DecNeedleForest",
         4: "DecBroadForest", 5: "MixedForest", 6: "ClosedShrub", 7: "OpenShrub",
         8: "WoodySavanna", 9: "Savanna", 10: "Grassland", 11: "Wetland",
         12: "Cropland", 13: "Urban", 14: "CropMosaic", 15: "Snow/Ice",
         16: "Barren", 17: "Water", 18: "WoodyTundra", 19: "MixedTundra",
         20: "BarrenTundra"}


def sample_vegtype(err):
    """NEAREST-sample static vgtyp to the error file's stations (categorical)."""
    vg = xr.open_zarr(HRRR_ZARR)["vgtyp"].isel(t0=0).squeeze(drop=True).load()
    ds_in = xr.Dataset({"vg": (("y", "x"), vg.values)}, coords={
        "lat": (("y", "x"), vg["latitude"].values),
        "lon": (("y", "x"), vg["longitude"].values % 360.0)})
    ds_out = xr.Dataset(coords={"lat": ("station", err["latitude"].values),
                                "lon": ("station", err["longitude"].values)})
    rg = xesmf.Regridder(ds_in, ds_out, method="nearest_s2d", locstream_out=True)
    vg_st = np.rint(rg(ds_in["vg"]).values).astype(int)
    return pd.Series(vg_st, index=err["station"].values)


def per_class_table(df):
    """Count-weighted per-class error summary for one gap frame."""
    rows = []
    for k, s in df.groupby("vg"):
        if len(s) < MIN_PER_CLASS:
            continue
        w = s["w"].values
        rows.append(dict(
            vg=k, name=NAMES.get(k, str(k)), n=len(s),
            med_elev=float(np.median(s["orography"])),
            only_logRMSE=np.average(np.log10(s["rmse_gap"]), weights=w),
            hrrr_logRMSE=np.average(np.log10(s["rmse_ref"]), weights=w),
            gap=np.average(s["log_ratio"], weights=w)))
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def main():
    rmse_ref, count_ref, topo = eg.load_model("gfs-hrrr")
    rmse_gap, count_gap, _ = eg.load_model("gfs-only")
    err = xr.open_dataset(f"{HD}/spatial.rmse.convobs.nested-lam.nc")
    vmap = sample_vegtype(err)

    tables = {}
    for var in ("2m_temperature", "10m_wind_speed"):
        df = eg.build_gap_frame(var, 0, rmse_gap, count_gap, rmse_ref, count_ref, topo)
        df["vg"] = df["station"].map(vmap)
        t = per_class_table(df)
        tables[var] = t
        print(f"\n=== {var} (fhr 0) by HRRR vegetation type, sorted by gap ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # bar plot of the gap by class for both variables
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, (var, t) in zip(axes, tables.items()):
        ax.barh(t["name"], t["gap"], color="tab:red")
        ax.set_xlabel("gap = mean log10(RMSE) gfs-only - gfs-hrrr")
        ax.set_title(var)
        ax.invert_yaxis()
        ax.axvline(0, color="0.4", lw=1)
    fig.suptitle("Error gap by HRRR vegetation type (fhr 0)", fontsize=13)
    out = "./veg_gap_by_class.png"
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
