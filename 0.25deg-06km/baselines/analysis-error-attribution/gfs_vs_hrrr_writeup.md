# GFS vs HRRR analysis differences at observation sites

**What and why.** We had been scoring GFS-vs-obs and HRRR-vs-obs separately, and the
attribution got convoluted. Since both analyses (the `fhr=0` initial conditions) are
scored against the *same* conventional surface obs, their **difference** is directly
informative: it is the part of the obs-space error gap that is purely a GFS↔HRRR
*representation* difference, with the observations cancelling out. We evaluate that
difference at the obs station locations — a meaningful, terrain-following sample of the
domain — and ask how much of it is explained by how differently the two models resolve
the terrain.

**Bottom line.** The 2 m temperature difference is, in complex terrain, a clean
**elevation/lapse-rate artifact**; over the flat ~80% of stations it is small and not
terrain-driven. The 10 m wind-speed difference is **not** explained by terrain at all.
The temporal variance of the difference is the *dominant* term of the mean-square error
but tracks terrain only weakly, and HRRR does **not** resolve systematically more
temporal variability than GFS.

---

## Data and method

| | |
|---|---|
| Stations | 3,585 conventional-obs sites inside the HRRR domain (of 21,826 finite-location sites) |
| Period | test year, 2024-02-01 → 2025-01-31 (293 analysis cycles, 30-h-spaced `t0`, so all hours of day are sampled) |
| Fields | `2m_temperature`; `10m_wind_speed` = √(u10²+v10²), formed on each native grid before interpolation |
| Sign convention | **HRRR − GFS** everywhere (positive ⇒ HRRR larger/warmer/windier/higher) |

- GFS analyses: `baselines/gfs-forecasts-vs-gfs-analysis/gfs.forecasts.zarr` (regular 0.25° grid).
- HRRR analyses: `baselines/hrrr-forecasts-vs-hrrr-analysis/hrrr.forecasts.zarr` (curvilinear LCC).
- Orography (`orog`, absent from the forecast zarrs): the `case-studies/la-fires/{gfs,hrrr}.forecasts.zarr` files, whose grids match the forecast grids exactly.
- Interpolation: one `xesmf` bilinear locstream regridder **per model**, built once and reused for every cycle and for the terrain sampling. The in-domain station mask is the bilinear-coverage ≥ 0.99 test against the HRRR grid.

**Per-station quantities** (over the 293 cycles), for each field `d = HRRR − GFS`:

- `bias` = mean_t(d)
- `rmse` = √(mean_t(d²))
- `var_diff` = var_t(d) — error variance; **MSE = bias² + var_diff** (verified exact)
- `var_gfs`, `var_hrrr` = each model's own temporal variance
- `mean_gfs`, `mean_hrrr`, `count`

The static terrain difference each model sees at each station: surface elevation
`orog_{gfs,hrrr,diff}` and the native-grid slope amplitude |∇h| `slope_{gfs,hrrr,diff}`.

**Outputs:** `$SCRATCH/nested-eagle/0.25deg-06km/production/gfs-vs-hrrr/`
→ `gfs_vs_hrrr.metrics.nc`, `gfs_vs_hrrr.topo.nc`, `gfs_vs_hrrr.roughness.nc`.

---

## Domain-mean summary

| Field | bias | rmse | var_diff | var_gfs | var_hrrr | MSE |
|---|---|---|---|---|---|---|
| 2 m T | +0.25 K | 1.95 K | 3.40 K² | 100.9 K² | 104.5 K² | 4.11 K² |
| 10 m wind | −0.38 m/s | 1.28 m/s | 1.31 | 3.98 | 4.03 | 1.76 |

Terrain: HRRR sits **34 m lower** than GFS on average at these stations (std 84 m) and
is marginally steeper (+2.7 m/km). The negative mean reflects obs sites sitting in
valleys that the coarse GFS grid fills in.

---

## Finding 1 — 2 m T difference is an elevation/lapse-rate artifact (in complex terrain)

- `bias` vs `orog_diff`: OLS slope **−6.32 K/km** (robust Theil-Sen **−5.76 K/km**), i.e.
  where HRRR places the ground lower than GFS, its 2 m analysis is correspondingly
  warmer, at very nearly the average-atmosphere lapse rate (6.5 K/km).
- Maps (`terrain_maps_*`): the bias and `orog_diff` light up the **same** western
  mountain stations; removing the single fitted lapse term shrinks the bias spread
  0.80 → 0.60 K std (44% of the *global* bias variance, and substantially more within
  the high-relief West, where essentially all the signal lives — the East is flat).

### Read the correlation conditionally on terrain

The global Pearson `r = −0.66` **overstates** the relationship and must not be reported
on its own. The data is extremely concentrated and the correlation is carried by a
sparse, high-leverage tail:

| stratum | share | Pearson r | Spearman ρ | OLS slope |
|---|---|---|---|---|
| core `\|Δz\| < 50 m` | 79% | −0.09 | −0.10 | −3.0 K/km |
| mid `50–150 m` | 14% | −0.28 | −0.28 | −3.8 K/km |
| tail `\|Δz\| > 150 m` | 7% | −0.82 | −0.80 | −6.2 K/km |
| **full sample** | 100% | **−0.66** | **−0.35** | **−6.3 K/km** |

- 63% of stations have `|Δz| < 25 m`; dropping just the top **5%** of `|Δz|` stations
  collapses `r` from 0.66 → 0.32.
- The **Pearson ≫ Spearman** gap (0.66 vs 0.35) is the fingerprint of that leverage.

**This sharpens rather than refutes the story.** Two different things are being
conflated: the *slope* (physical, stable) and the *correlation coefficient* (depends on
the spread of x). OLS and the robust Theil-Sen slope agree (−6.3 vs −5.8 K/km), so the
slope is not an outlier artifact. The weak *core* correlation is **range restriction**:
across ±50 m the lapse signal is only ≈ ±0.3 K, smaller than the bias scatter from other
sources (diurnal sampling, model/obs representativeness), so `r` is crushed even though
the slope is intact. The flat-terrain core stations have ≈ zero elevation difference
*and* ≈ zero bias — consistent with the line, sitting at the origin, contributing no
leverage.

**Statement to use:** *the GFS↔HRRR 2 m temperature difference is a lapse-rate/elevation
artifact where the terrain actually differs (complex western terrain), where it is
strong (ρ ≈ −0.8) and physically calibrated (slope ≈ lapse rate). For the ~80% of
flat-terrain stations the elevation difference is negligible and the small residual bias
has other causes.* Lead with the binned median, Spearman ρ, and robust slope; keep
Pearson/OLS only for reference.

---

## Finding 2 — 10 m wind difference: a robust but modest roughness tendency, not terrain

Terrain barely explains the wind difference: `bias` vs terrain only `r ≈ −0.28`
(elevation) / `+0.29` (slope); `rmse` vs `|terrain diff|` `r ≈ 0.05`. The systematic
driver is **surface roughness** — but the honest framing is *robust modest tendency*:
roughness explains very little at any single station, while displaying a clear general
trend across the population.

The zarrs carry no roughness, so we pulled the true aerodynamic roughness length
(`SFCR`, z0) for both models from the public NOAA GRIB archives
(`noaa-gfs-bdp-pds`, `noaa-hrrr-bdp-pds`), byte-ranging the single `SFCR` message per
cycle via its `.idx`. Roughness is quasi-static but seasonal, so we sampled one cycle
per month over the test year and averaged `log10(z0)`; the predictor is
`rough_diff = log10(z0_hrrr) − log10(z0_gfs)` (>0 ⇒ HRRR rougher). Comparison is in log
space because the models define z0 differently (GFS z0 up to ~2.3 m, HRRR up to ~0.8 m).

- `bias` vs `rough_diff`: **r = −0.42, ρ = −0.43** — where HRRR is rougher than GFS its
  10 m wind is slower (negative bias), the physically correct sign.
- **Strongest single wind predictor** (ρ = −0.43 vs terrain −0.23 / +0.13). Joint
  standardized regression `bias ~ rough_diff + slope_diff + orog_diff`: β = −0.39 /
  +0.20 / −0.14, **R² = 0.26** (roughness alone R² = 0.18 — more than double terrain).
- `var_diff` vs `|rough_diff|` ρ = 0.26, also beating terrain.

### It survives the leverage critique that the T story did not

The scatter looks at first like "a vertical blob with a regression line dragged through
outliers." It is not. Unlike Finding 1, this relationship is **robust, not
leverage-driven**, by three independent checks (`roughness_diagnostic_*`):

| stratum | share | Pearson r | Spearman ρ |
|---|---|---|---|
| core `\|rd\| < 0.1` | 23% | −0.21 | −0.23 |
| mid `0.1–0.3` | 39% | −0.27 | −0.22 |
| `0.3–0.7` | 28% | −0.42 | −0.41 |
| tail `\|rd\| > 0.7` | 10% | −0.72 | −0.69 |
| **full** | 100% | **−0.42** | **−0.43** |

1. **Present in every stratum, core included** (core r = −0.21 with a *steep* slope) —
   contrast the T core's r ≈ −0.09. The signal is not confined to a tail.
2. **Pearson ≈ Spearman throughout** (−0.42 vs −0.43). The rank correlation is immune to
   x-outliers, so its near-equality with Pearson is direct proof this is not leverage —
   the exact opposite of the T case (0.66 vs 0.35).
3. **Survives outlier deletion.** Dropping the top 5 / 10 / 20% of `|rough_diff|`
   stations leaves r at −0.36 / −0.34 / −0.31 (mild attenuation from lost x-range, no
   collapse). The T case collapsed 0.66 → 0.32 on dropping just the top 5%.
   The roughness distribution is genuinely broad (only 23% of stations within
   `|rd| < 0.1`, vs 79% within `|Δz| < 50 m` for elevation), so there is real x-spread
   to fit rather than a spike at zero.

### But it is modest — a trend, not a law

r = −0.42 ⇒ **R² ≈ 0.18**: roughness accounts for only ~18% of the station-to-station
wind-bias variance. The other ~82% is the vertical thickness of the blob (diurnal
sampling, local representativeness, etc.), so **at any individual station roughness
explains very little** and is not a usable per-site predictor. What is real and robust
is the **central tendency**: the binned-median spine descends monotonically from
~+0.3 m/s at the smooth end to ~−0.6 m/s at the rough end (`roughness_diagnostic_*`,
left panel). The within-stratum slope also saturates (steep −1.7 in the core, shallow
−0.37 in the tail), so the binned median — not a single linear slope — is the right
summary.

**Statement to use:** *where HRRR is rougher than GFS, its 10 m wind is systematically
slower — a robust but modest population tendency (ρ ≈ −0.43, survives outlier deletion,
present within every stratum). It explains very little at any one location (R² ≈ 0.18)
but is a clear general trend in the data, and the strongest single predictor of the wind
difference, well ahead of terrain.* Not "roughness explains the wind bias."

Caveat: z0 definitions differ between the two models, so `rough_diff` mixes a genuine
representation difference with a definitional one — the sign and robustness are credible,
the magnitude calibration is not; the floor (`Z0_FLOOR=1e-4 m`) inflates `|rough_diff|`
at a few coast/water stations (handled by log-space and robust stats).

---

## Finding 3 — Variance: dominant in MSE, weak on terrain, and a clean null on "more variability"

- **`var_diff` is the dominant MSE term.** For 2 m T, mean `var_diff` ≈ 3.4 K² is ~83%
  of the 4.1 K² MSE; the mean `bias²` is only ~17%. So the clean lapse-rate attribution
  explains the *mean offset*, which is the **small** part — most of the disagreement is
  time-varying scatter.
- **`var_diff` tracks terrain only modestly:** 2 m T vs `|orog_diff|` ρ = +0.31; wind
  ρ = +0.20 (vs `|slope_diff|` weaker). Real but well below the bias's lapse signal.
- **HRRR does not resolve more temporal variability.** `var_hrrr/var_gfs` ≈ 1 everywhere
  (median 1.05 for T, 1.01 for wind) and does **not** track terrain (T `log`-ratio vs
  `|slope_diff|` ρ = −0.10; wind ρ = +0.03; the wind Pearson r ≈ +0.38 is a few outlier
  stations, not a trend). A worthwhile null for the paper.

---

## Scripts and figures

Pipeline (in `baselines/analysis-error-attribution/`, run in the `eagle` conda env;
heavy step ran on an interactive CPU node):

| script | produces |
|---|---|
| `compare_gfs_hrrr.py` | the two `.nc` metric/topo files |
| `fetch_roughness.py` | pulls GFS/HRRR `SFCR` from NOAA GRIB → `gfs_vs_hrrr.roughness.nc` (`rough_diff`) |
| `plot_roughness.py` | `roughness_relation_*.png`, `roughness_maps_*.png` — wind bias vs roughness |
| `plot_roughness_diagnostic.py` | `roughness_diagnostic_*.png` — median spine + IQR ribbon, and Pearson/Spearman vs outlier deletion (answers the "blob through outliers" critique) |
| `plot_terrain_relation.py` | `terrain_relation_gfs_vs_hrrr.png` — bias/rmse vs terrain (linear-count) |
| `plot_terrain_relation_logcount.py` | `terrain_relation_logcount_gfs_vs_hrrr.png` — bias vs terrain, **log-count** density + robust/Spearman annotations (shows the leverage explicitly) |
| `plot_terrain_maps.py` | `terrain_maps_gfs_vs_hrrr.png` — CONUS orog_diff │ T bias │ lapse-removed residual |
| `plot_variance.py` | `variance_terrain_gfs_vs_hrrr.png`, `variance_ratio_maps_gfs_vs_hrrr.png` |

Notes:
- `6.5 K/km` reference is the average-atmosphere (ICAO standard-atmosphere) lapse rate —
  a yardstick, **not** the dry-adiabatic (9.8) or moist (~5) rate.
- Map colormaps are discrete 7-bin diverging (3−/near-zero/3+) with `extend='both'`.
