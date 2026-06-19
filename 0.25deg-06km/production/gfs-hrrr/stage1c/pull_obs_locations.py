"""
Pull observation *locations* (no data values) over a random 24h period.

Reuses the obs side of the obs_metrics workflow (eagle.tools.obs_metrics) to
load conventional observations from the same datastreams / quantities used by
obs-metrics.global.testing.yaml, QC-filters them, and writes a flat NetCDF
table of where (and when) observations exist for each quantity / datastream.

Output schema (single ``obs`` dimension):
    LAT(obs), LON(obs), time(obs)
    datastream(obs)   e.g. conv-adpsfc-NC000007
    quantity(obs)     e.g. 2m_temperature

One row per (observation, quantity-it-reports). A station that reports
temperature, wind, and pressure contributes three rows (the "long"/melted
layout). Locations are kept only where the obs passes QC (max_qc_value).

Usage:
    conda run -n eagle python pull_obs_locations.py
    conda run -n eagle python pull_obs_locations.py --config obs-metrics.global.testing.yaml
    conda run -n eagle python pull_obs_locations.py --start 2024-07-15T00 --seed 0
"""
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import nnja_ai

from eagle.tools.obs_metrics import (
    _load_obs_config,
    build_variable_map,
    _build_rename_map,
    _load_one_dataset,
    apply_qc_filter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pull_obs_locations")


def pick_random_day(start_date, end_date, seed=None):
    """Pick a random full UTC calendar day (24h) within [start_date, end_date].

    nnja selects observations by daily partitions, so the natural 24h unit is a
    full UTC day. Returns (day_midnight, next_midnight).
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    n_days = int((end - start) / pd.Timedelta("1D"))
    if n_days < 1:
        raise ValueError("Date range too short for a 24h window")

    rng = np.random.default_rng(seed)
    offset_days = int(rng.integers(0, n_days))  # exclude last day so it fits fully
    win_start = start + pd.Timedelta(days=offset_days)
    win_end = win_start + pd.Timedelta("24h")
    return win_start, win_end


def build_quantity_validity_columns(variable_map):
    """Map each base quantity -> set of columns whose non-NaN value marks a
    valid observation of that quantity.

    - wind components -> the wind-speed source column (obs_wspd_col)
    - derived wind speed -> obs_col already points at the wspd column
    - everything else  -> obs_col
    Upper-air quantities contribute one column per level; a location counts as
    observed if *any* level is valid.
    """
    quantity_cols = defaultdict(set)
    for vinfo in variable_map.values():
        bn = vinfo["base_name"]
        if "obs_wspd_col" in vinfo:
            quantity_cols[bn].add(vinfo["obs_wspd_col"])
        else:
            quantity_cols[bn].add(vinfo["obs_col"])
    return quantity_cols


def collect_locations(time_range, variable_map, dataset_registry, max_qc_value):
    """Load each datastream, QC-filter, and emit a long-form locations frame."""
    quantity_cols = build_quantity_validity_columns(variable_map)
    dc = nnja_ai.DataCatalog()

    win_start, win_end = time_range
    # nnja selects by daily partitions; keep the slice inside the day so only
    # the target partition loads, then filter to the exact [start, end) window.
    nnja_range = (win_start, win_end - pd.Timedelta("1min"))

    frames = []
    for dataset_name, registry in dataset_registry.items():
        rename_map = _build_rename_map(registry, variable_map)
        if not rename_map:
            continue

        obs_df = _load_one_dataset(dc, dataset_name, rename_map, nnja_range)
        if obs_df is None or len(obs_df) == 0:
            continue

        # Guarantee obs fall inside the requested 24h window (nnja is daily-grained)
        ts = pd.to_datetime(obs_df["OBS_TIMESTAMP"])
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        in_window = (ts.to_numpy() >= np.datetime64(win_start)) & (
            ts.to_numpy() < np.datetime64(win_end)
        )
        obs_df = obs_df.loc[in_window]
        if len(obs_df) == 0:
            continue

        # Mask obs/wspd values to NaN where QC > max_qc_value
        obs_df = apply_qc_filter(obs_df, variable_map, max_qc_value=max_qc_value)

        # Which base quantities does this datastream actually provide?
        ds_quantities = {registry_key for registry_key in registry}

        for quantity, cols in quantity_cols.items():
            if quantity not in ds_quantities:
                continue
            present = [c for c in cols if c in obs_df.columns]
            if not present:
                continue

            valid = obs_df[present].notna().any(axis=1)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            sub = obs_df.loc[valid, ["LAT", "LON", "OBS_TIMESTAMP"]].copy()
            sub["datastream"] = dataset_name
            sub["quantity"] = quantity
            frames.append(sub)
            logger.info(
                f"{dataset_name}: {n_valid:>8d} valid locations for {quantity}"
            )

    if not frames:
        logger.warning("No observation locations found in the window.")
        return pd.DataFrame(
            columns=["LAT", "LON", "OBS_TIMESTAMP", "datastream", "quantity"]
        )

    return pd.concat(frames, ignore_index=True)


def to_dataset(loc_df, time_range, config_path, max_qc_value):
    """Convert the long-form locations frame to an xr.Dataset on dim 'obs'."""
    time = pd.to_datetime(loc_df["OBS_TIMESTAMP"])
    if getattr(time.dt, "tz", None) is not None:
        time = time.dt.tz_convert("UTC").dt.tz_localize(None)

    xds = xr.Dataset(
        data_vars={
            "LAT": ("obs", loc_df["LAT"].to_numpy(np.float64)),
            "LON": ("obs", loc_df["LON"].to_numpy(np.float64)),
            "time": ("obs", time.to_numpy()),
            "datastream": ("obs", loc_df["datastream"].to_numpy(object)),
            "quantity": ("obs", loc_df["quantity"].to_numpy(object)),
        },
        coords={"obs": np.arange(len(loc_df))},
    )
    xds["LAT"].attrs = {"units": "degrees_north", "long_name": "observation latitude"}
    xds["LON"].attrs = {
        "units": "degrees_east",
        "long_name": "observation longitude (native datastream convention)",
    }
    xds.attrs = {
        "title": "Conventional observation locations (no data values)",
        "description": (
            "Locations/times of QC-passed observations over a 24h period, "
            "one row per (observation, reported quantity)."
        ),
        "source_config": config_path,
        "window_start": str(time_range[0]),
        "window_end": str(time_range[1]),
        "max_qc_value": int(max_qc_value),
    }
    return xds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="obs-metrics.global.testing.yaml",
        help="Path to the obs-metrics yaml (for variables, dates, QC).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Override the random day (e.g. 2024-07-15). UTC; normalized to midnight.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random window choice."
    )
    parser.add_argument(
        "--output", default="obs_locations.nc", help="Output NetCDF path."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    obs_config = _load_obs_config()
    dataset_registry = obs_config["dataset_registry"]
    levels = config.get("levels", None)
    max_qc_value = config.get("max_qc_value", 2)

    variable_map = build_variable_map(config, levels, obs_config=obs_config)
    logger.info(f"Quantities requested: {sorted({v['base_name'] for v in variable_map.values()})}")
    logger.info(f"Datastreams: {list(dataset_registry.keys())}")

    if args.start is not None:
        win_start = pd.Timestamp(args.start).normalize()
        win_end = win_start + pd.Timedelta("24h")
    else:
        win_start, win_end = pick_random_day(
            config["start_date"], config["end_date"], seed=args.seed
        )
    logger.info(f"24h window (full UTC day): {win_start} -> {win_end}")

    loc_df = collect_locations(
        (win_start, win_end), variable_map, dataset_registry, max_qc_value
    )
    logger.info(f"Total location rows: {len(loc_df)}")

    xds = to_dataset(loc_df, (win_start, win_end), args.config, max_qc_value)
    xds.to_netcdf(args.output)
    logger.info(f"Wrote {args.output}")
    logger.info(f"\n{xds}")


if __name__ == "__main__":
    main()
