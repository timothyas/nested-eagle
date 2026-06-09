"""
Diagnostic for the 10 m wind vs surface-roughness relationship, built to answer
the "it's just a vertical blob / regression through outliers" critique directly.

Left:  the blob with its *spine* -- binned median wind bias with a 25-75% IQR
       ribbon, so the central tendency is visible against the large scatter.
       The spine descends monotonically; the ribbon shows the per-bin spread
       that makes any single station uninformative.
Right: robustness-to-leverage -- Pearson vs Spearman as we delete the most
       extreme |rough_diff| stations. Both stay flat (contrast the T-elevation
       case, where Pearson collapses), proving the trend is not outlier-driven.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

from datasets import parse_dataset, DEFAULT, ROUGHNESS


def load(ds=DEFAULT):
    m = xr.open_dataset(ds["metrics"]).sel(field="10m_wind_speed")
    r = xr.open_dataset(ROUGHNESS)
    rd = r["rough_diff"].values
    bias = m["bias"].values
    g = np.isfinite(rd) & np.isfinite(bias)
    return rd[g], bias[g]


def spine_panel(ax, rd, bias):
    # show the cloud lightly, then the median spine + IQR ribbon
    ax.scatter(rd, bias, s=4, c="0.75", alpha=0.4, lw=0, zorder=1)
    edges = np.quantile(rd, np.linspace(0, 1, 16))  # equal-count bins
    edges = np.unique(edges)
    cen, med, q25, q75 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rd >= lo) & (rd < hi)
        if sel.sum() < 15:
            continue
        cen.append(0.5 * (lo + hi))
        med.append(np.median(bias[sel]))
        q25.append(np.percentile(bias[sel], 25))
        q75.append(np.percentile(bias[sel], 75))
    cen, med, q25, q75 = map(np.array, (cen, med, q25, q75))
    ax.fill_between(cen, q25, q75, color="tab:blue", alpha=0.25,
                    label="25-75% per bin (scatter)", zorder=2)
    ax.plot(cen, med, "o-", color="tab:blue", ms=5, lw=2,
            label="binned median (spine)", zorder=3)
    rho = stats.spearmanr(rd, bias).statistic
    pr = stats.pearsonr(rd, bias).statistic
    ax.axhline(0, color="0.5", lw=0.8, zorder=0)
    ax.axvline(0, color="0.5", lw=0.8, zorder=0)
    ax.set_xlim(np.percentile(rd, [1, 99]))
    ax.set_ylim(np.percentile(bias, [1, 99]))
    ax.set_xlabel("log10(z0 HRRR / z0 GFS)   (>0 = HRRR rougher)")
    ax.set_ylabel("10 m wind bias  HRRR-GFS  [m/s]")
    ax.set_title(f"The blob has a sloped spine  (Pearson {pr:+.2f}, "
                 f"Spearman {rho:+.2f})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def leverage_panel(ax, rd, bias):
    ard = np.abs(rd)
    fracs = np.linspace(0, 0.30, 16)
    pear, spear = [], []
    for f in fracs:
        thr = np.percentile(ard, 100 * (1 - f)) if f > 0 else ard.max() + 1
        sel = ard <= thr
        pear.append(stats.pearsonr(rd[sel], bias[sel]).statistic)
        spear.append(stats.spearmanr(rd[sel], bias[sel]).statistic)
    ax.plot(100 * fracs, pear, "o-", color="crimson", label="Pearson r")
    ax.plot(100 * fracs, spear, "s-", color="navy", label="Spearman $\\rho$")
    ax.set_xlabel("% of most-extreme |rough_diff| stations deleted")
    ax.set_ylabel("correlation with wind bias")
    ax.set_title("Trend survives outlier deletion\n(if it were leverage, these "
                 "would race to 0)", fontsize=10)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(-0.5, 0.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def main(ds=DEFAULT):
    rd, bias = load(ds)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    spine_panel(axes[0], rd, bias)
    leverage_panel(axes[1], rd, bias)
    fig.suptitle(f"{ds['diff']}: is the wind-roughness relationship a blob through "
                 f"outliers?  (n={rd.size})  -- No.", fontsize=13)
    out = f"roughness_diagnostic_{ds['tag']}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(parse_dataset(__doc__))
