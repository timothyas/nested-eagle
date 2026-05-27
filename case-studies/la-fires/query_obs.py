"""
Query NNJA-AI METAR/SPECI data for Southern California ASOS stations
during the January 2025 Santa Ana wind / LA fires event.

Uses NNJA-AI dataset `conv-adpsfc-NC000007` (ADP Surface — Aviation METAR/SPECI),
which carries US METAR reports keyed by ICAO identifier.

Requires: pip install "nnja-ai[complete]>=1.0.0"

Output columns (renamed from raw `MTR*` BUFR descriptors):
  - OBS_TIMESTAMP : observation time (UTC)
  - RPID          : ICAO station identifier
  - LAT, LON, SELV: position + station elevation (m)
  - WSPD          : sustained wind speed (m/s)
  - WDIR          : wind direction (deg)
  - MXGS          : max gust speed (m/s)
  - PKWDSP, PKWDDR: peak wind speed (m/s) and direction (deg)
"""

import pandas as pd
from nnja_ai import DataCatalog

DATASET_ID = "conv-adpsfc-NC000007"

# ── Target ASOS stations ──────────────────────────────────────────────
# Tier A: Pass / foothill stations (strong Santa Ana signal)
# Tier B: Coastal / valley contrast stations (weaker signal)
STATIONS = {
    # Tier A — Pass / Foothill / inland valley
    "KVNY": {"name": "Van Nuys",                "lat": 34.210, "lon": -118.490, "tier": "A"},
    "KBUR": {"name": "Burbank/Glendale",        "lat": 34.201, "lon": -118.359, "tier": "A"},
    "KEMT": {"name": "El Monte",                "lat": 34.086, "lon": -118.035, "tier": "A"},
    "KPOC": {"name": "Brackett Field/La Verne", "lat": 34.092, "lon": -117.782, "tier": "A"},
    "KONT": {"name": "Ontario Intl",            "lat": 34.056, "lon": -117.601, "tier": "A"},
    # Tier B — Coastal
    "KLAX": {"name": "Los Angeles Intl",        "lat": 33.943, "lon": -118.408, "tier": "B"},
    "KHHR": {"name": "Hawthorne Muni",          "lat": 33.923, "lon": -118.335, "tier": "B"},
    "KTOA": {"name": "Torrance Muni",           "lat": 33.803, "lon": -118.340, "tier": "B"},
    "KLGB": {"name": "Long Beach",              "lat": 33.818, "lon": -118.152, "tier": "B"},
}

# ── SoCal bounding box for broader spatial queries ────────────────────
SOCAL_BBOX = {
    "lat_min": 33.5,
    "lat_max": 34.8,
    "lon_min": -119.0,
    "lon_max": -117.0,
}

# ── Time range ───────────────────────────────────────────────────────
# Event: Jan 7-8, 2025. Extend window to capture pre/post event.
DATE_START = "2025-01-03"
DATE_END = "2025-01-12"

# ── Wind speed thresholds for onset detection (m/s) ───────────────────
THRESHOLDS_MS = {
    "red_flag": 12.0,       # ~27 mph, Red Flag Warning level
    "wind_advisory": 15.0,  # ~34 mph, Wind Advisory level
    "high_wind": 18.0,      # ~40 mph, High Wind Warning level
}

# ── NNJA variables to request, and the friendly column names we expose ─
NNJA_VARS = [
    "OBS_TIMESTAMP",
    "LAT",
    "LON",
    "SELV",
    "MTRID.RPID",
    "MTRWND.WDIR",
    "MTRWND.WSPD",
    "MTRWND.MTGUST.MXGS",
    "MTRPKW.PKWDSP",
    "MTRPKW.PKWDDR",
]

RENAME = {
    "MTRID.RPID":          "RPID",
    "MTRWND.WDIR":         "WDIR",
    "MTRWND.WSPD":         "WSPD",
    "MTRWND.MTGUST.MXGS":  "MXGS",
    "MTRPKW.PKWDSP":       "PKWDSP",
    "MTRPKW.PKWDDR":       "PKWDDR",
}


def load_metar_for_dates(
    date_start: str,
    date_end: str,
    variables: list[str] = NNJA_VARS,
) -> pd.DataFrame:
    """Load METAR/SPECI data for a range of dates from NNJA-AI."""
    catalog = DataCatalog(mirror="gcp_brightband")
    dataset = catalog[DATASET_ID]

    dates = pd.date_range(date_start, date_end, freq="1D", tz="UTC")
    frames = []

    for date in dates:
        print(f"Loading {date.strftime('%Y-%m-%d')}...")
        try:
            ds_day = (
                dataset
                .sel(time=date, variables=variables)
                .load_dataset(backend="pandas")
            )
            frames.append(ds_day)
        except Exception as e:
            print(f"  Warning: could not load {date.strftime('%Y-%m-%d')}: {e}")

    if not frames:
        raise RuntimeError("No data loaded for the specified date range.")

    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns=RENAME)


def filter_by_stations(df: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    """Filter DataFrame to specific ICAO station IDs."""
    return df[df["RPID"].isin(station_ids)].copy()


def filter_by_bbox(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """Filter DataFrame to a geographic bounding box."""
    mask = (
        (df["LAT"] >= lat_min)
        & (df["LAT"] <= lat_max)
        & (df["LON"] >= lon_min)
        & (df["LON"] <= lon_max)
    )
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════
# Main script
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print(f"Loading METAR data from NNJA-AI ({DATASET_ID})...")
    print("=" * 60)
    df_all = load_metar_for_dates(DATE_START, DATE_END)
    print(f"\nLoaded {len(df_all):,} total METAR observations.")

    print("\nFiltering to Southern California bounding box...")
    df_socal = filter_by_bbox(df_all, **SOCAL_BBOX)
    print(f"  {len(df_socal):,} observations in SoCal bbox "
          f"({df_socal['RPID'].nunique()} unique stations).")

    icao_ids = list(STATIONS.keys())
    df_stations = filter_by_stations(df_socal, icao_ids).sort_values("OBS_TIMESTAMP")
    print(f"\nTarget ASOS stations: {len(df_stations):,} obs.")

    print("\n" + "=" * 60)
    print("Summary by station:")
    print("=" * 60)
    for icao, meta in STATIONS.items():
        stn = df_stations[df_stations["RPID"] == icao]
        if len(stn) == 0:
            print(f"  {icao} ({meta['name']}): no data")
            continue
        max_wspd = stn["WSPD"].max()
        max_gust = stn["MXGS"].max()
        print(
            f"  {icao} ({meta['name']}, Tier {meta['tier']}): "
            f"{len(stn)} obs, "
            f"max sustained={max_wspd:.1f} m/s ({max_wspd * 2.237:.0f} mph), "
            f"max gust={max_gust:.1f} m/s ({max_gust * 2.237:.0f} mph)"
        )

    print("\n" + "=" * 60)
    print("Onset detection (first time sustained wind exceeds threshold):")
    print("=" * 60)
    for thresh_name, thresh_val in THRESHOLDS_MS.items():
        print(f"\n  Threshold: {thresh_name} ({thresh_val} m/s / "
              f"{thresh_val * 2.237:.0f} mph)")
        for icao, meta in STATIONS.items():
            stn = df_stations[df_stations["RPID"] == icao]
            exceed = stn[stn["WSPD"] >= thresh_val]
            if len(exceed) > 0:
                first = exceed["OBS_TIMESTAMP"].min()
                print(f"    {icao}: first exceedance at {first}")
            else:
                print(f"    {icao}: never exceeded")

    outfile = "socal_wind_obs_jan2025.parquet"
    df_stations.to_parquet(outfile, index=False)
    print(f"\nSaved station data to {outfile}")

    outfile_bbox = "socal_bbox_obs_jan2025.parquet"
    df_socal.to_parquet(outfile_bbox, index=False)
    print(f"Saved SoCal bbox data to {outfile_bbox}")

    print("\nDone!")
