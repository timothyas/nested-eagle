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
temporal variability than GFS. Finally, when two ML models inherit these analyses
(Nested-EAGLE vs Global-EAGLE), the **2 m-T elevation/lapse artifact persists into their
24 h forecast essentially unchanged, while the 10 m-wind roughness effect is inherited but
decays by roughly half.** A temporal decomposition recovers part of that dominant variance
as a **robust diurnal+seasonal cycle (~32% for 2 m T, ~10% for wind)**; the rest is
synoptic.

---

## Data and method

| | |
|---|---|
| Stations | 3,585 conventional-obs sites inside the HRRR domain (of 21,826 finite-location sites) |
| Period | test year, 2024-02-01 → 2025-01-31 (293 analysis cycles, 30-h-spaced `t0`, so all hours of day are sampled) |
| Fields | `2m_temperature`; `10m_wind_speed` = √(u10²+v10²), formed on each native grid before interpolation |
| Sign convention | **HRRR − GFS** everywhere (positive → HRRR larger/warmer/windier/higher) |

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
`orog_{gfs,hrrr,diff}` and the native-grid slope amplitude |grad h| `slope_{gfs,hrrr,diff}`.

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

## Terrain dependence at a glance

We correlate `bias` against *signed* terrain (direction matters) and `var_diff` against
`|terrain|` (magnitude), three ways: **Pearson on the raw predictor**, **Pearson after
log-scaling it** (`sign(x)·log10(1+|x|)` for bias, `log10|x|` for var), and **Spearman ρ**
(rank-based — identical under any monotonic transform, so it is the leverage-free anchor).

| field | pair | Pearson raw | Pearson log | Spearman |
|---|---|---:|---:|---:|
| 2 m T | bias ~ elev | −0.66 | −0.37 | −0.35 |
| 2 m T | bias ~ slope | +0.47 | +0.36 | +0.27 |
| 2 m T | var ~ \|elev\| | +0.38 | +0.34 | +0.31 |
| 2 m T | var ~ \|slope\| | +0.18 | +0.19 | +0.15 |
| 10 m wind | bias ~ elev | −0.28 | −0.17 | −0.23 |
| 10 m wind | bias ~ slope | +0.29 | +0.16 | +0.13 |
| 10 m wind | var ~ \|elev\| | +0.19 | +0.23 | +0.20 |
| 10 m wind | var ~ \|slope\| | +0.13 | +0.15 | +0.11 |

**Reading key.** *raw >> log ≈ rank* → the raw correlation is **leverage-inflated** by the
sparse high-terrain tail (it deflates once the tail is compressed or the data ranked).
*raw ≈ log ≈ rank* → a **robust bulk** relationship, no transform or outlier responsible.

- Every **bias ~ terrain** row shows the leverage pattern (raw well above log ≈ rank).
  The T `bias ~ elev` collapse **−0.66 → −0.37 ≈ −0.35** is the cleanest one-number
  statement of the Finding 1 caveat.
- Every **var ~ terrain** row is robust but modest. `var ~ |elev|` (ρ ≈ 0.31 T, 0.20 wind)
  is the one terrain signal that holds up — and log is no better than raw, so it is
  **linear in magnitude, not logarithmic**.
- **Elevation beats slope** in every pairing.

(`plot_terrain_log.py` → `terrain_log_gfs_vs_hrrr.png`.)

---

## Finding 1 — 2 m T difference is an elevation/lapse-rate artifact (in complex terrain)

- `bias` vs `orog_diff`: OLS slope **−6.32 K/km** (robust Theil-Sen **−5.76 K/km**), i.e.
  where HRRR places the ground lower than GFS, its 2 m analysis is correspondingly
  warmer, at very nearly the average-atmosphere lapse rate (6.5 K/km).
- Maps (`terrain_maps_*`): the bias and `orog_diff` light up the **same** western
  mountain stations; removing the single fitted lapse term shrinks the bias spread
  0.80 → 0.60 K std (44% of the *global* bias variance, and substantially more within
  the high-relief West, where essentially all the signal lives — the East is flat).

**Read the correlation conditionally on terrain.** The raw Pearson `r = −0.66` overstates
it — the glance table shows it deflating to −0.37 ≈ Spearman −0.35 under log/rank, the
signature of a tail-carried correlation. Concretely, 79% of stations have `|Δz| < 50 m`
and within that flat core `r ≈ −0.09`; dropping just the top 5% of `|Δz|` collapses `r`
to 0.32. But the **slope is stable** (OLS −3 K/km in the core rising to −6.2 in the tail,
≈ Theil-Sen −5.8), so the relationship is physical, not an outlier artifact — the weak
*core* correlation is **range restriction** (±50 m carries only ≈ ±0.3 K of lapse signal,
below the bias scatter from diurnal sampling and representativeness).

*Rigor note.* Leverage is the **only** inflation here — unlike the temporal decomposition
(Finding 3 follow-on), this regression fits 2 parameters from 3,585 stations, so the
finite-sample/over-fit correction is nil (adjusted R² = raw R² = 0.440 to four decimals).
If anything the per-station bias, being a 293-cycle mean, carries sampling noise (~2% of
the across-station bias variance) that slightly *deflates* R². So the honest
elevation-explained share is the leverage-free bulk **ρ² ≈ 0.12**, not the tail-leveraged
0.44.

**Statement to use:** *a lapse-rate/elevation artifact where the terrain actually differs
(complex western terrain) — strong there (ρ ≈ −0.8) and physically calibrated
(slope ≈ lapse rate); for the ~80% of flat-terrain stations Δz ≈ 0 and the small residual
bias has other causes.* Lead with the binned median, Spearman ρ, and robust slope, not
raw Pearson.

---

## Finding 2 — 10 m wind difference: a robust but modest roughness tendency, not terrain

Terrain barely explains the wind difference (glance table: `bias ~ terrain` ρ ≈ −0.23 /
+0.13, all leverage-deflating; `rmse ~ |terrain|` r ≈ 0.05). The systematic driver is
**surface roughness** — though the honest framing is a *robust modest tendency*: it
explains very little at any one station but is a clear population trend.

The zarrs carry no roughness, so we pulled the true aerodynamic roughness length
(`SFCR`, z0) for both models from the NOAA GRIB archives (`noaa-gfs-bdp-pds`,
`noaa-hrrr-bdp-pds`), byte-ranging the single `SFCR` message per cycle via its `.idx`;
one cycle per month, averaged in `log10(z0)`. The predictor is
`rough_diff = log10(z0_hrrr) − log10(z0_gfs)` (>0 → HRRR rougher), compared in log space
because the models define z0 differently (GFS up to ~2.3 m, HRRR up to ~0.8 m).

- `bias` vs `rough_diff`: **r = −0.42, ρ = −0.43** — HRRR rougher → 10 m wind slower
  (correct sign). Strongest single wind predictor; joint standardized
  `bias ~ rough + slope + orog`: β = −0.39 / +0.20 / −0.14, **R² = 0.26** (roughness alone
  0.18, double terrain). `var_diff` vs `|rough_diff|` ρ = 0.26, also beating terrain.

**Robust, unlike the T bias.** The scatter looks like "a blob with a line dragged through
outliers," but three checks (`roughness_diagnostic_*`) say otherwise:

| stratum | share | Pearson r | Spearman ρ |
|---|---|---:|---:|
| core `\|rd\| < 0.1` | 23% | −0.21 | −0.23 |
| mid `0.1–0.3` | 39% | −0.27 | −0.22 |
| `0.3–0.7` | 28% | −0.42 | −0.41 |
| tail `\|rd\| > 0.7` | 10% | −0.72 | −0.69 |
| **full** | 100% | **−0.42** | **−0.43** |

(1) present in **every stratum**, core included (vs the T core's −0.09); (2)
**Pearson ≈ Spearman** throughout (the leverage-free signature, opposite the T case); (3)
**survives outlier deletion** — dropping the top 5/10/20% `|rd|` leaves r at
−0.36/−0.34/−0.31 (no collapse; the T case collapsed 0.66 → 0.32 on the top 5% alone).
Roughness is also genuinely spread (only 23% within `|rd| < 0.1`, vs 79% within
`|Δz| < 50 m`), so there is real x-range to fit.

**But modest — a trend, not a law.** R² ≈ 0.18: roughness explains ~18% of the
station-to-station wind-bias variance; the rest is the blob's vertical thickness (diurnal
sampling, representativeness). So it is a **central tendency, not a per-site predictor** —
the binned-median spine descends monotonically (~+0.3 → −0.6 m/s) and saturates (slope
−1.7 in the core, −0.37 in the tail), so the binned median, not a single slope, is the
right summary. *Rigor note.* As in Finding 1 this R² needs no finite-sample correction
(adjusted = raw to four decimals: 3,585 stations, one predictor) — and here raw R² ≈
robust ρ² ≈ 0.18 already, so not even leverage applies; the ~18% is honest bulk.

**Statement to use:** *where HRRR is rougher than GFS its 10 m wind is systematically
slower — robust but modest (ρ ≈ −0.43, survives outlier deletion, present in every
stratum), explaining little at any one site (R² ≈ 0.18) but a clear general trend and the
strongest single predictor.* Not "roughness explains the wind bias." Caveat: z0
definitions differ between the models, so `rough_diff` mixes a real representation
difference with a definitional one — sign and robustness credible, magnitude not.

---

## Finding 3 — Variance: dominant in MSE, weak on terrain, and a clean null on "more variability"

- **`var_diff` dominates the MSE.** For 2 m T, mean `var_diff` ≈ 3.4 K² is ~83% of the
  4.1 K² MSE (mean `bias²` only ~17%) — the clean lapse attribution explains the mean
  offset, which is the *small* part; most disagreement is time-varying scatter.
- **It tracks terrain only modestly, but robustly.** `var ~ |elevation|` is the one
  terrain signal that survives the raw/log/rank test (ρ ≈ 0.31 T, 0.20 wind; glance
  table) — a genuine bulk relationship, linear in magnitude, but well below the bias's
  lapse signal. Slope is weaker still.
- **HRRR does not resolve more temporal variability.** `var_hrrr/var_gfs` ≈ 1 everywhere
  (median 1.05 T, 1.01 wind) and does **not** track terrain (T `log`-ratio vs
  `|slope_diff|` ρ = −0.10; wind ρ = +0.03; the wind Pearson r ≈ +0.38 is a few outliers,
  not a trend). A worthwhile null for the paper.

---

## Finding 4 — Does the signature survive into the ML forecast? (T yes, wind partly)

The findings above are about the *analysis* difference (`fhr=0`). The natural follow-up:
do two ML models that ingested these analyses **inherit** the surface signature, and does
it survive 24 h of forecast evolution? We repeat the *identical* regression on the 24 h
forecast difference of two models — **Nested-EAGLE** (HRRR over CONUS + GFS elsewhere)
minus **Global-EAGLE** (GFS only) — at the same stations, with the **same predictors**
(the terrain/roughness each model sees *is* the HRRR−GFS difference: Nested inherits HRRR,
Global inherits GFS). Sign convention Nested − Global, mirroring HRRR − GFS. (3,255 of the
3,585 stations fall inside the trimmed Nested-LAM grid; the rest are dropped.)

**2 m T — the elevation/lapse artifact is inherited almost perfectly.**

| | lapse fit | r | var removed | `bias~elev` raw/log/rank |
|---|---|---|---|---|
| analysis `fhr=0` (HRRR−GFS) | −6.32 K/km | −0.66 | 44% | −0.66 / −0.37 / −0.35 |
| **forecast `fhr=24`** (Nested−Global) | **−6.58 K/km** | **−0.67** | **45%** | **−0.67 / −0.37 / −0.34** |

Essentially unchanged across every diagnostic — slope, correlation, the leverage
fingerprint (raw >> log ≈ rank), and the ~45% removable by a single lapse term. The
elevation/lapse representation difference is carried wholesale by the ML models and
survives 24 h intact. `var ~ |elev|` stays a robust bulk signal too (rank 0.31 → 0.27).

**10 m wind — the roughness effect is inherited but decays by ~half.**

| | roughness slope | r | var removed | `bias~elev` raw/log/rank |
|---|---|---|---|---|
| analysis `fhr=0` | −0.42 m/s per dex z0 | −0.42 | 18% | −0.28 / −0.17 / −0.23 |
| **forecast `fhr=24`** | **−0.24 m/s per dex z0** | **−0.28** | **8%** | −0.19 / −0.09 / −0.16 |

The roughness imprint is still present with the correct sign, but roughly halved. This is
physically sensible: the *analysis* 10 m wind is tightly slaved to the static surface
roughness, whereas the 24 h forecast wind is increasingly set by the evolving
boundary-layer and synoptic state, diluting the static-surface signal. The mean
differences track the analysis too (+0.25 K / −0.38 m/s → +0.31 K / −0.45 m/s).

**Variance still dominates and stays unexplained.** As at `fhr=0`, `var_diff` is the bulk
of the MSE at `fhr=24` (80% for 2 m T, 64% for 10 m wind) and the best terrain/roughness
predictor explains ≤7% of it (`var~|elev|` ρ=0.27 T; `var~|rough|` ρ=0.19 wind). So the
clean bias attributions still account for only the small mean-offset term; the dominant
time-varying scatter remains largely unaccounted for. The var-fraction does fall slightly
into the forecast (wind 74→64%) because `var_diff` itself shrinks faster than `bias²` — the
two models' forecasts agree more in their temporal scatter than the two analyses did — but
variance still wins.

**Statement to use:** *the 2 m-T elevation/lapse-rate artifact is a wholesale inheritance
that persists into the 24 h ML forecast essentially unchanged; the 10 m-wind
surface-roughness effect is inherited with the correct sign but washes out roughly halfway
by 24 h as forecast dynamics take over from the static surface imprint. In both the
analysis and the forecast, the time-varying variance — not the attributed mean bias —
remains the dominant and still-unexplained share of the MSE.*

---

## Finding 5 — Temporal decomposition of the variance (diurnal/seasonal, not terrain)

If terrain barely explains `var_diff`, what does? We decompose each station's per-cycle
difference `d(t0)` by **time** — a two-way ANOVA over month (season) × hour-of-day
(diurnal), the 293 cycles falling into 12 months × {00,06,12,18} UTC = 48 cells.

*Statistics first.* With only ~6 cycles per cell, raw η² is badly inflated — 48 cell-means
explain ≈ (48−1)/(293−1) ≈ **0.16 of the variance by chance alone** (confirmed by a label
permutation null). We therefore report the **bias-corrected ε² (= adjusted R²)**, which is
effect-size-aware and recovers the true share at any magnitude (verified by simulation).
Significance is *not* the limiter — with 3,585 stations everything is formally significant;
effect size is. The corrected shares of `var_diff`:

| component | 2 m T | 10 m wind |
|---|---:|---:|
| season (month) | 0.12 | 0.04 |
| diurnal (hour) | **0.16** | 0.04 |
| interaction | 0.04 | ~0.01 |
| **combined (det.)** | **0.32** | **0.10** |
| residual (synoptic) | **0.68** | **0.90** |

- **2 m T:** a robust **diurnal** cycle (ε²≈0.16, F≈23 — only 4 well-sampled groups, so
  trustworthy) plus a **seasonal** cycle (≈0.12); together ~32%, with RMS ≈ 1.2 K vs the
  synoptic residual's 1.4 K — *comparable*. The diurnal share concentrates over the
  **interior/mountain West** (arid, high-insolation, deep boundary layers), where the two
  models' surface/BL/radiation schemes diverge most on the daily cycle.
- **10 m wind:** the deterministic clock is weak (~10%); the difference is **~90% synoptic**.
- **Interaction caveat:** the season×hour term (a season-modulated diurnal cycle) is, after
  correction, ~0.04 for T and ~0 for wind — at the edge of what 293 cycles can resolve.
  Raw η² made it look ~0.12; that was mostly the finite-sample floor. Pinning it down needs
  the full hourly archive, not 30 h-spaced cycles.

**Statement to use:** *the variance terrain could not explain is partly explainable after
all — but by time, not place: a robust diurnal+seasonal cycle (~32% for 2 m T, concentrated
where the boundary-layer physics differ; ~10% for wind). The majority (≈68% T, ≈90% wind)
is genuinely synoptic, consistent with the dynamics-driven persistence of Finding 4.*

---

## Scripts and figures

Pipeline (in `baselines/analysis-error-attribution/`, run in the `eagle` conda env;
heavy step ran on an interactive CPU node):

| script | produces |
|---|---|
| `compare_gfs_hrrr.py` | the two `.nc` metric/topo files (HRRR−GFS analysis, `fhr=0`) |
| `compare_nested_global.py` | `nested_vs_global.fhr24.metrics.nc` — Nested−Global 24 h forecast difference at the same stations (Finding 4); runs on an interactive node |
| `datasets.py` | dataset descriptors + the `--dataset {gfs_vs_hrrr,nested_vs_global}` flag every plot script takes (default `gfs_vs_hrrr`; same predictors, swapped metrics/labels) |
| `fetch_roughness.py` | pulls GFS/HRRR `SFCR` from NOAA GRIB → `gfs_vs_hrrr.roughness.nc` (`rough_diff`) |
| `decompose_variance.py` | `variance_decomp_*.png` + table — season/diurnal/interaction/residual shares of `var_diff` (Finding 5); reads the per-cycle `*.diffs.nc` |
| `variance_significance.py` | finite-sample honesty: raw η² vs permutation null vs bias-corrected ε² (adjusted R²) + F, the basis for Finding 5's numbers |
| `plot_roughness.py` | `roughness_relation_*.png`, `roughness_maps_*.png` — wind bias vs roughness |
| `plot_roughness_diagnostic.py` | `roughness_diagnostic_*.png` — median spine + IQR ribbon, and Pearson/Spearman vs outlier deletion (answers the "blob through outliers" critique) |
| `plot_terrain_relation.py` | `terrain_relation_gfs_vs_hrrr.png` — bias/rmse vs terrain (linear-count) |
| `plot_terrain_relation_logcount.py` | `terrain_relation_logcount_gfs_vs_hrrr.png` — bias vs terrain, **log-count** density + robust/Spearman annotations (shows the leverage explicitly) |
| `plot_terrain_log.py` | `terrain_log_gfs_vs_hrrr.png` — bias/var vs **log-scaled** terrain; the raw/log/rank table (leverage vs robust diagnostic) |
| `plot_terrain_maps.py` | `terrain_maps_gfs_vs_hrrr.png` — CONUS orog_diff │ T bias │ lapse-removed residual |
| `plot_variance.py` | `variance_terrain_gfs_vs_hrrr.png`, `variance_ratio_maps_gfs_vs_hrrr.png` |

Each plot script takes `--dataset nested_vs_global` to regenerate its figure for the
Finding 4 (`fhr=24`) comparison; outputs are tagged `*_nested_vs_global.fhr24.png`. The
`gfs_vs_hrrr` default reproduces the analysis-study figures unchanged.

Notes:
- `6.5 K/km` reference is the average-atmosphere (ICAO standard-atmosphere) lapse rate —
  a yardstick, **not** the dry-adiabatic (9.8) or moist (~5) rate.
- Map colormaps are discrete 7-bin diverging (3−/near-zero/3+) with `extend='both'`.
