"""
Plot observed wind speeds (sustained + gust) at the SoCal ASOS stations
during the January 2025 Santa Ana / LA fires event.

One subplot per station, sharing the time axis.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from query_obs import STATIONS, THRESHOLDS_MS

INFILE = "socal_wind_obs_jan2025.parquet"
OUTFILE = "socal_wind_timeseries.png"

MS_TO_MPH = 2.237


def main():
    df = pd.read_parquet(INFILE)
    df["OBS_TIMESTAMP"] = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True)
    df = df.sort_values("OBS_TIMESTAMP")

    stations = list(STATIONS.keys())
    n = len(stations)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(11, 1.7 * n),
        sharex=True,
        constrained_layout=True,
    )

    for ax, icao in zip(axes, stations):
        meta = STATIONS[icao]
        stn = df[df["RPID"] == icao]

        if len(stn) == 0:
            ax.text(0.5, 0.5, f"{icao}: no data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
            continue

        t = stn["OBS_TIMESTAMP"]
        ax.plot(t, stn["WSPD"], color="C0", lw=1.0,
                label="sustained")
        ax.plot(t, stn["MXGS"], color="C3", lw=0.8, alpha=0.7,
                label="gust")

        for name, val in THRESHOLDS_MS.items():
            ax.axhline(val, color="gray", lw=0.5, ls="--", alpha=0.6)

        ax.set_ylabel("m/s")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{icao} — {meta['name']} (Tier {meta['tier']})",
            loc="left", fontsize=10,
        )

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("Date (UTC), January 2025")

    fig.suptitle(
        "SoCal ASOS observed wind speed — Jan 2025 Santa Ana event",
        fontsize=12,
    )

    fig.savefig(OUTFILE, dpi=150)
    print(f"Saved {OUTFILE}")


if __name__ == "__main__":
    main()
