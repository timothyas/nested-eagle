"""
Finite-sample honesty check for the temporal variance decomposition.

Raw eta^2 (= SS_between/SS_total) is upward-biased: estimating k group means from N
samples explains a fraction ~ (k-1)/(N-1) of the variance even when the grouping is
random. With N~293 cycles, the marginals (12 months, 4 hours) carry small bias, but the
48-cell month x hour model carries E[eta^2|null] ~ 47/292 ~ 0.16 -- a big slice of the
raw "combined" number.

We therefore report, per field, aggregated over stations:
  - raw eta^2                (biased)
  - permutation-null eta^2   (shuffle t0<->(month,hour) labels; the empirical bias floor)
  - bias-corrected epsilon^2 (subtract df_X * MS_residual; the honest effect size)
and an F-test (with so many station-cycles, significance is never the issue -- effect
size and its bias are).
"""
import numpy as np
import xarray as xr
from scipy import stats

from datasets import parse_dataset, DEFAULT
from decompose_variance import load, FIELDS

KEYS = {"season": ("month", 11), "diurnal": ("hour", 3), "combined": ("cell", 47)}
NPERM = 24


def ss_between(d, dbar, key):
    grp = d.groupby(key)
    mean = grp.mean()
    cnt = grp.count()
    gdim = [k for k in mean.dims if k != "station"][0]
    return (cnt * (mean - dbar) ** 2).sum(gdim)


def agg_fracs(d, dbar, ss_total_sum, good):
    """Aggregate eta^2 for each component over good stations."""
    out = {}
    for name, (key, _) in KEYS.items():
        out[name] = float(ss_between(d, dbar, key).where(good).sum() / ss_total_sum)
    return out


def main(ds=DEFAULT):
    d_all = load(ds)
    rng = np.random.default_rng(0)
    print(f"N cycles = {d_all.sizes['t0']}   cells = 12 months x 4 hours = 48   "
          f"(~{d_all.sizes['t0']/48:.1f} per cell)\n")

    for f, (lab, unit) in FIELDS.items():
        d = d_all.sel(field=f)
        month = d["t0"].dt.month.values
        hour = d["t0"].dt.hour.values
        d = d.assign_coords(month=("t0", month), hour=("t0", hour),
                            cell=("t0", month * 100 + hour))
        dbar = d.mean("t0")
        n = d.count("t0")
        ss_total = ((d - dbar) ** 2).sum("t0")
        good = np.isfinite(ss_total) & (ss_total > 0)
        sst = float(ss_total.where(good).sum())

        raw = agg_fracs(d, dbar, sst, good)
        raw["interaction"] = raw["combined"] - raw["season"] - raw["diurnal"]

        # bias-corrected epsilon^2: subtract df_X * MS_resid (full 48-cell model resid)
        ss_cell = ss_between(d, dbar, "cell")
        ss_resid = (ss_total - ss_cell).where(good)
        df_resid = (n - 48).where(good)            # per station
        ms_resid = float(ss_resid.sum() / df_resid.sum())
        ngood = int(good.sum())
        eps = {}
        for name, (key, dfx) in KEYS.items():
            ss_x = float(ss_between(d, dbar, key).where(good).sum())
            eps[name] = (ss_x - ngood * dfx * ms_resid) / sst
        eps["interaction"] = eps["combined"] - eps["season"] - eps["diurnal"]

        # permutation null: shuffle which t0 carries which (month,hour) label
        null = {k: [] for k in KEYS}
        for _ in range(NPERM):
            p = rng.permutation(d.sizes["t0"])
            dp = d.assign_coords(month=("t0", month[p]), hour=("t0", hour[p]),
                                 cell=("t0", (month * 100 + hour)[p]))
            for name, (key, _) in KEYS.items():
                null[name].append(
                    float(ss_between(dp, dbar, key).where(good).sum() / sst))
        nullm = {k: np.mean(v) for k, v in null.items()}
        nullm["interaction"] = (nullm["combined"] - nullm["season"]
                                - nullm["diurnal"])

        DF = {"season": 11, "diurnal": 3, "interaction": 33, "combined": 47}
        print(f"{lab}:")
        print(f"   {'component':12s} {'raw eta2':>9s} {'null eta2':>10s} "
              f"{'corrected':>10s} {'F':>8s}")
        for name in ("season", "diurnal", "interaction", "combined"):
            F = (raw[name] * sst / (ngood * DF[name])) / ms_resid
            print(f"   {name:12s} {raw[name]:9.3f} {nullm[name]:10.3f} "
                  f"{eps[name]:10.3f} {F:8.1f}")
        print(f"   (null eta2 ~ (k-1)/(N-1); corrected = bias-subtracted effect size; "
              f"F>>1 = significant)\n")


if __name__ == "__main__":
    main(parse_dataset(__doc__))
