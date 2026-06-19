"""
Map of the observation stations used in the case study, over terrain.

Stations are colored by their data-driven tier (see query_obs.STATIONS):
  Tier A  — strong Santa Ana signal (peak 3 h-mean sustained >= 12 m/s)
  Tier B  — control, did not see it (peak 3 h-mean sustained <  8 m/s)
  dropped — ambiguous / out-of-regime / poor obs (shown faint for context)

The Palisades and Eaton fire ignition points are marked for reference.
Terrain (HRRR orography) is shaded to show the San Gabriel / Santa Monica
mountains that channel the flow.
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eagle.tools.data import open_forecast_zarr_dataset
from query_obs import STATIONS

SCRATCH = os.environ["SCRATCH"]
LA_FIRES = os.path.join(SCRATCH, "nested-eagle/case-studies/la-fires")
HRRR_ZARR = os.path.join(LA_FIRES, "hrrr.forecasts.zarr")
OBS_FILE = "socal_bbox_obs_jan2025.parquet"

# Map extent (covers all stations + terrain context)
_WEST, _EAST, _SOUTH, _NORTH = -118.75, -117.45, 33.70, 34.45
OROG_LEVELS = np.arange(0, 2600, 250)

# Fire ignition points
FIRES = {
    "Palisades": (34.0736, -118.5547),
    "Eaton":     (34.1897, -118.1314),
}

# Tier styling
TIER_STYLE = {
    "A":    dict(color="#c1272d", marker="o", s=90, label="Tier A — strong (≥12 m/s)"),
    "B":    dict(color="#0061a8", marker="s", s=80, label="Tier B — control (<8 m/s)"),
    "drop": dict(color="0.55",    marker="x", s=55, label="dropped (excluded)"),
}

OBS_WIN = ("2025-01-06", "2025-01-11")


def peak_3h_mean() -> dict:
    """Peak 3 h-centered-mean sustained wind per station over the event."""
    df = pd.read_parquet(OBS_FILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    out = {}
    for icao in STATIONS:
        g = df[(df["RPID"] == icao)
               & (df["OBS_TIMESTAMP"] >= OBS_WIN[0])
               & (df["OBS_TIMESTAMP"] < OBS_WIN[1])]
        if len(g) == 0:
            out[icao] = np.nan
            continue
        s = (g.set_index("OBS_TIMESTAMP")["WSPD"].sort_index()
             .resample("30min").mean()
             .rolling("3h", center=True, min_periods=1).mean())
        out[icao] = float(s.max())
    return out


def load_orog():
    ds = open_forecast_zarr_dataset(
        path=HRRR_ZARR, t0="2025-01-06T12",
        vars_of_interest=["orog"], load=True, reshape_cell_to_2d=True,
    ).squeeze()
    lat = ds["latitude"].values
    lon = np.where(ds["longitude"].values > 180,
                   ds["longitude"].values - 360, ds["longitude"].values)
    return lat, lon, ds["orog"].values


def draw_stations(ax, peaks=None, label=True, label_peak=False,
                  fires=True, fire_label=None, fontsize=7.5):
    """Overlay tier-colored station markers (+ ICAO labels) and fire ignition
    points on a cartopy axis. Reusable across figures. White text halos keep
    labels legible over a filled background (e.g. a wind field)."""
    proj = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=2, foreground="white")]
    if fire_label is None:
        fire_label = label

    if fires:
        for name, (flat, flon) in FIRES.items():
            ax.scatter(flon, flat, marker="*", s=320, color="#ff7f0e",
                       edgecolors="black", linewidths=0.8, zorder=6, transform=proj)
            if fire_label:
                ax.annotate(f"{name} Fire", (flon, flat), xytext=(6, -12),
                            textcoords="offset points", fontsize=fontsize,
                            style="italic", color="#cc5500", transform=proj,
                            zorder=7, path_effects=halo)

    for tier, style in TIER_STYLE.items():
        members = [s for s in STATIONS if STATIONS[s]["tier"] == tier]
        if not members:
            continue
        lons = [STATIONS[s]["lon"] for s in members]
        lats = [STATIONS[s]["lat"] for s in members]
        ax.scatter(lons, lats, transform=proj,
                   edgecolors="black" if style["marker"] != "x" else style["color"],
                   linewidths=0.7, zorder=5,
                   **{k: v for k, v in style.items() if k != "label"},
                   label=style["label"])
        if not label:
            continue
        for s in members:
            txt = s
            if label_peak and peaks is not None and np.isfinite(peaks.get(s, np.nan)):
                txt = f"{s}\n{peaks[s]:.1f}"
            ax.annotate(txt, (STATIONS[s]["lon"], STATIONS[s]["lat"]),
                        xytext=(6, 5), textcoords="offset points",
                        fontsize=fontsize, fontweight="bold",
                        color="black" if tier != "drop" else "0.5",
                        transform=proj, zorder=7, path_effects=halo)


def main():
    plt.style.use(os.path.expandvars("$HOME/nice.mplstyle"))

    print("Computing peak 3 h-mean winds...")
    peaks = peak_3h_mean()
    print("Loading HRRR orography...")
    lat_o, lon_o, orog = load_orog()

    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    ax.set_extent([_WEST, _EAST, _SOUTH, _NORTH], crs=ccrs.PlateCarree())

    # Terrain
    cf = ax.contourf(
        lon_o, lat_o, orog, levels=OROG_LEVELS, cmap=cmo.gray_r,
        alpha=0.6, transform=ccrs.PlateCarree(), extend="max", zorder=0,
    )
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.03,
                        pad=0.02, shrink=0.8)
    cbar.set_label("Elevation (m)")

    ax.add_feature(cfeature.COASTLINE, lw=0.8, edgecolor="black")
    ax.add_feature(cfeature.STATES, lw=0.4, edgecolor="gray")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeaf3", zorder=-1)

    draw_stations(ax, peaks=peaks, label=True, label_peak=True, fires=True, fontsize=7.5)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, title="Observation stations")
    ax.set_title(
        "Case 1 observation stations (label: ICAO / peak 3 h-mean sustained wind, m/s)\n"
        "Jan 6–11 2025 — tiers set by peak 3 h-mean wind (A ≥ 12, B < 8 m/s)",
        loc="left", fontsize=11,
    )

    os.makedirs("figures", exist_ok=True)
    outfile = "figures/station_map.png"
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
