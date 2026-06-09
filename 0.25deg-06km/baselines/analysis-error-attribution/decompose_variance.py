"""
Temporal decomposition of var_diff -- the dominant, terrain-unexplained MSE term.

Findings 3/4 showed that var_diff = var_t(d), where d(t0) = A - B is the per-cycle
difference (HRRR - GFS analysis, or Nested - Global forecast), is ~64-83% of the MSE and
that terrain/roughness explains <=9% of it. Here we ask a different question: how much of
var_diff is the *deterministic* seasonal (time-of-year) and diurnal (time-of-day) cycle of
the difference, versus an unstructured synoptic residual?

Per station, one field, with N finite cycles d_i and grand mean dbar:

  SS_total = sum_i (d_i - dbar)^2                          (= N * var_diff)
  SS_month = sum_m  n_m  (mean_m  - dbar)^2                seasonal cycle (12 groups)
  SS_hour  = sum_h  n_h  (mean_h  - dbar)^2                diurnal cycle (hours sampled)
  SS_cell  = sum_mh n_mh (mean_mh - dbar)^2                full month x hour structure
  eta2_X   = SS_X / SS_total                               fraction of var explained
  residual = 1 - eta2_cell                                 synoptic / unstructured

The interaction SS_cell - SS_month - SS_hour is ~0 when the month and hour sampling are
orthogonal (they nearly are at the 30 h t0 spacing, which cycles hour-of-day through
00/06/12/18 independently of the slow seasonal drift).

Reported two ways:
  - variance-weighted (domain aggregate):  sum_s SS_X,s / sum_s SS_total,s
  - per-station median eta2 (the typical station)

Outputs:
  - stdout: explained-variance table per field
  - variance_decomp_<tag>.png : per-station maps of eta2_season and eta2_diurnal
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datasets import parse_dataset, DEFAULT

FIELDS = {"2m_temperature": ("2 m T", "K"), "10m_wind_speed": ("10 m wind", "m/s")}
EXTENT = [-125, -66, 23, 50]  # CONUS


def load(ds):
    d = xr.open_dataset(ds["diffs"])["diff"]
    d = d.assign_coords(month=d["t0"].dt.month, hour=d["t0"].dt.hour)
    return d


def ss_between(d, dbar, key):
    """Between-group sum of squares: sum_g n_g (mean_g - dbar)^2, per station."""
    grp = d.groupby(key)
    mean = grp.mean()                       # (group, station), skips NaN
    cnt = grp.count()                       # n_g per station
    gdim = [k for k in mean.dims if k != "station"][0]
    return (cnt * (mean - dbar) ** 2).sum(gdim)


def decompose(d):
    """Return SS_total and the season/diurnal/combined between-SS, per station."""
    dbar = d.mean("t0")                      # grand mean (skipna)
    ss_total = ((d - dbar) ** 2).sum("t0")
    d = d.assign_coords(cell=d["month"] * 100 + d["hour"])
    return {
        "total": ss_total,
        "season": ss_between(d, dbar, "month"),
        "diurnal": ss_between(d, dbar, "hour"),
        "combined": ss_between(d, dbar, "cell"),
    }


def report_table(d):
    """Print, per field, each component's explained FRACTION of var_diff (eta^2,
    dimensionless) alongside its absolute variance (field units^2) and RMS amplitude
    (field units), so 'modest fraction' can be read against the physical scale."""
    out = {}
    for f, (lab, unit) in FIELDS.items():
        df = d.sel(field=f)
        ss = decompose(df)
        n = df.count("t0")
        tot = ss["total"]
        good = np.isfinite(tot) & (tot > 0)

        # variance-weighted aggregate fraction of var_diff explained
        frac = {k: float(ss[k].where(good).sum() / tot.where(good).sum())
                for k in ("season", "diurnal", "combined")}
        # interaction = month x hour structure not captured by the two marginals
        # (a season-modulated diurnal cycle); residual = synoptic / unstructured
        frac["interaction"] = frac["combined"] - frac["season"] - frac["diurnal"]
        frac["residual"] = 1.0 - frac["combined"]
        # domain-mean component variance (field units^2) and RMS amplitude (field units)
        var = {k: float((ss[k] / n).where(good).mean())
               for k in ("total", "season", "diurnal", "combined")}
        var["interaction"] = var["combined"] - var["season"] - var["diurnal"]
        var["residual"] = var["total"] - var["combined"]

        print(f"\n{lab}  (n={int(good.sum())} stations):  "
              f"var_diff = {var['total']:.3f} {unit}^2  (RMS {np.sqrt(var['total']):.2f} {unit})")
        print(f"   {'component':11s} {'fraction':>9s} {'variance':>13s} {'RMS':>10s}")
        for k in ("season", "diurnal", "interaction", "combined", "residual"):
            tag = "= season+diurnal+interaction" if k == "combined" else ""
            print(f"   {k:11s} {frac[k]:9.3f} {var[k]:9.3f} {unit}^2 "
                  f"{np.sqrt(max(var[k], 0)):7.2f} {unit}  {tag}")
        out[f] = ss
    return out


def smap(ax, lon, lat, c, vmax, title, label):
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, lw=0.3, edgecolor="0.6")
    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    sc = ax.scatter(lon, lat, c=c, s=9, cmap="viridis", vmin=0, vmax=vmax,
                    transform=ccrs.PlateCarree(), edgecolor="none")
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.03, shrink=0.9,
                      extend="max")
    cb.set_label(label, fontsize=9)


def maps_figure(d, ss_by_field, ds):
    lon = ((d["longitude"].values + 180) % 360) - 180
    lat = d["latitude"].values
    fig, axes = plt.subplots(len(FIELDS), 2, figsize=(14, 5.6 * len(FIELDS)),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    for i, (f, (lab, _)) in enumerate(FIELDS.items()):
        ss = ss_by_field[f]
        tot = ss["total"].where(np.isfinite(ss["total"]) & (ss["total"] > 0))
        es = (ss["season"] / tot).values
        ed = (ss["diurnal"] / tot).values
        vmax = float(np.nanpercentile(np.concatenate([es, ed]), 98))
        smap(axes[i, 0], lon, lat, es, vmax, f"{lab}: seasonal $\\eta^2$",
             "fraction of var_diff (season)")
        smap(axes[i, 1], lon, lat, ed, vmax, f"{lab}: diurnal $\\eta^2$",
             "fraction of var_diff (time of day)")
    fig.suptitle(f"Temporal decomposition of var({ds['diff']}) {ds['kind']}: "
                 "seasonal vs diurnal share per station", fontsize=13)
    out = f"variance_decomp_{ds['tag']}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def main(ds=DEFAULT):
    d = load(ds)
    print(f"diffs: {dict(d.sizes)}  hours sampled: "
          f"{sorted(set(int(h) for h in np.unique(d['hour'].values)))}\n")
    ss_by_field = report_table(d)
    maps_figure(d, ss_by_field, ds)


if __name__ == "__main__":
    main(parse_dataset(__doc__))
