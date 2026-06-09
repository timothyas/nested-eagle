"""
Dataset descriptors shared by the regression/plot scripts.

Two parallel analyses regress the SAME terrain/roughness predictors against a
per-station HRRR-minus-GFS-style difference:

  gfs_vs_hrrr       : HRRR   - GFS     analysis initial condition (fhr=0)
  nested_vs_global  : Nested - Global  24 h ML forecast          (fhr=24)

The predictors are identical for both, because the terrain/roughness each ML model
effectively sees is exactly the analysis difference: Nested uses HRRR over CONUS,
Global uses GFS. So orog_diff/slope_diff (gfs_vs_hrrr.topo.nc) and rough_diff
(gfs_vs_hrrr.roughness.nc) are reused as-is, and the only thing that changes between
the two analyses is the metrics file and the labels.

Every plot script takes ``--dataset`` (default gfs_vs_hrrr, so the paper figures are
unchanged) and pulls its metrics path, model labels, sign string, and output-name tag
from here -- one code path over either difference.
"""
import argparse
import os

DATA = f"{os.environ['SCRATCH']}/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr"

# Shared predictors (same terrain/roughness for both analyses).
TOPO = f"{DATA}/gfs_vs_hrrr.topo.nc"
ROUGHNESS = f"{DATA}/gfs_vs_hrrr.roughness.nc"

DATASETS = {
    "gfs_vs_hrrr": dict(
        metrics=f"{DATA}/gfs_vs_hrrr.metrics.nc",
        tag="gfs_vs_hrrr",
        a="hrrr", b="gfs",                 # var_<a>/var_<b>, mean_<a>/mean_<b>
        a_label="HRRR", b_label="GFS",
        diff="HRRR - GFS",
        kind="analysis (fhr=0)",
    ),
    "nested_vs_global": dict(
        metrics=f"{DATA}/nested_vs_global.fhr24.metrics.nc",
        tag="nested_vs_global.fhr24",
        a="nested", b="global",
        a_label="Nested-EAGLE", b_label="Global-EAGLE",
        diff="Nested - Global",
        kind="24 h forecast (fhr=24)",
    ),
}

DEFAULT = DATASETS["gfs_vs_hrrr"]


def parse_dataset(doc=None):
    """Return the descriptor selected by ``--dataset`` (default gfs_vs_hrrr)."""
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="gfs_vs_hrrr", choices=list(DATASETS),
                   help="which per-station difference to analyze")
    return DATASETS[p.parse_args().dataset]
