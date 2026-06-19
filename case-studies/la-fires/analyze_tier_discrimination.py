"""
Does each model discriminate Tier A (strong) from Tier B (control) stations?

For every model and station, compute the PEAK sustained 10 m wind over the
event window (max over the forecast valid times), using the same bilinear
point sampling as the time-series figure; for the high-res models also the
5 km neighborhood max. Aggregate over inits, then report the Tier A vs Tier B
contrast (mean A − mean B) per model and compare it to the observed contrast.

The headline question: is Nested-EAGLE's Tier-A peak meaningfully above its
Tier-B peak, the way the observations are? If the contrast collapses, the
resolution advantage isn't translating into a usable signal.
"""

import numpy as np
import pandas as pd
import xesmf as xe

from plot_wind_early_detection import (
    load_eagle, load_global_eagle, load_hrrr, load_gfs,
    interp_lcc, interp_gfs, _station_target, build_masks, envelope,
    EAGLE_GFS_INITS, HRRR_INITS, MODELS, HIRES_MODELS,
    TARGET_STATIONS, OBS_FILE, OBS_RESAMPLE, OBS_WINDOW,
)
from query_obs import STATIONS

WIN = slice("2025-01-06", "2025-01-11")
A = [s for s in TARGET_STATIONS if STATIONS[s]["tier"] == "A"]
B = [s for s in TARGET_STATIONS if STATIONS[s]["tier"] == "B"]


def obs_peaks() -> dict:
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    out = {}
    for s in TARGET_STATIONS:
        g = df[(df["RPID"] == s)
               & (df["OBS_TIMESTAMP"] >= "2025-01-06")
               & (df["OBS_TIMESTAMP"] < "2025-01-11")]
        ser = (g.set_index("OBS_TIMESTAMP")["WSPD"].sort_index()
               .resample(OBS_RESAMPLE).mean()
               .rolling(OBS_WINDOW, center=True, min_periods=1).mean())
        out[s] = float(ser.max())
    return out


def main():
    target = _station_target()
    point = {m: {s: [] for s in TARGET_STATIONS} for m in MODELS}
    nbhd  = {m: {s: [] for s in TARGET_STATIONS} for m in HIRES_MODELS}
    eagle_rg = hrrr_rg = None
    eagle_masks = hrrr_masks = None

    def stash(store, model, da):
        for s in TARGET_STATIONS:
            store[model][s].append(float(da.sel(station=s, time=WIN).max()))

    print("Loading EAGLE / Global-EAGLE / GFS...")
    for t0 in EAGLE_GFS_INITS:
        de = load_eagle(t0)
        if eagle_rg is None:
            eagle_rg = xe.Regridder(de, target, "bilinear", locstream_out=True)
            eagle_masks = build_masks(de)
        stash(point, "Nested-EAGLE", interp_lcc(de, eagle_rg))
        _, hi = envelope(de, eagle_masks)
        stash(nbhd, "Nested-EAGLE", hi)
        stash(point, "Global-EAGLE", interp_gfs(load_global_eagle(t0)))
        stash(point, "GFS", interp_gfs(load_gfs(t0)))

    print("Loading HRRR...")
    for t0 in HRRR_INITS:
        dh = load_hrrr(t0)
        if hrrr_rg is None:
            hrrr_rg = xe.Regridder(dh, target, "bilinear", locstream_out=True)
            hrrr_masks = build_masks(dh)
        stash(point, "HRRR", interp_lcc(dh, hrrr_rg))
        _, hi = envelope(dh, hrrr_masks)
        stash(nbhd, "HRRR", hi)

    # Aggregate per station: mean (and max) of per-init peaks
    def agg(store, model):
        return {s: (np.mean(store[model][s]), np.max(store[model][s]))
                for s in TARGET_STATIONS}

    obs = obs_peaks()

    print("\n" + "=" * 78)
    print("PEAK sustained wind per station  (obs 3h-mean; models = mean[max] of "
          "per-init peaks)")
    print("=" * 78)
    hdr = f"{'stn':6}{'tier':5}{'obs':>7}" + "".join(f"{m:>16}" for m in MODELS)
    print(hdr)
    pt = {m: agg(point, m) for m in MODELS}
    for s in TARGET_STATIONS:
        row = f"{s:6}{STATIONS[s]['tier']:5}{obs[s]:7.1f}"
        for m in MODELS:
            mean, mx = pt[m][s]
            row += f"{mean:7.1f}[{mx:4.1f}]"
        print(row)

    def tier_means(d):
        return np.mean([d[s][0] for s in A]), np.mean([d[s][0] for s in B])

    print("\n" + "=" * 78)
    print("TIER CONTRAST  (mean peak, point sampling)")
    print("=" * 78)
    oa, ob = np.mean([obs[s] for s in A]), np.mean([obs[s] for s in B])
    print(f"{'OBS':16} A={oa:6.1f}  B={ob:6.1f}  A-B={oa-ob:6.1f}  A/B={oa/ob:4.2f}")
    for m in MODELS:
        a, b = tier_means(pt[m])
        print(f"{m:16} A={a:6.1f}  B={b:6.1f}  A-B={a-b:6.1f}  A/B={a/b:4.2f}")

    print("\nNEIGHBORHOOD (5 km max) contrast, high-res models:")
    nb = {m: agg(nbhd, m) for m in HIRES_MODELS}
    for m in HIRES_MODELS:
        a, b = tier_means(nb[m])
        print(f"{m:16} A={a:6.1f}  B={b:6.1f}  A-B={a-b:6.1f}  A/B={a/b:4.2f}")


if __name__ == "__main__":
    main()
