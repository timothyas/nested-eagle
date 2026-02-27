# Trainable Parameters Test

How much do the graph-specific trainable parameters matter?
These need to be turned off for pretraining with different datasets, so it's
worth understanding what these actually bring to the table.

* `eight.yaml` The status quo thus far: eight trainable parameters on each of
  the graph stages
* `zero.yaml` all graph trainable parameters are set to zero

## Model size

With 8 trainable parameters turned on everywhere:

```
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name    ┃ Type                 ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model   │ AnemoiModelInterface │ 63.1 M │ train │     0 │
│ 1 │ loss    │ ModuleDict           │      0 │ train │     0 │
│ 2 │ metrics │ ModuleDict           │      0 │ train │     0 │
└───┴─────────┴──────────────────────┴────────┴───────┴───────┘
Trainable params: 63.1 M
Non-trainable params: 0
Total params: 63.1 M
Total estimated model params size (MB): 252
Modules in train mode: 302
Modules in eval mode: 0
Total FLOPs: 0
```

With them turned off:

```
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name    ┃ Type                 ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model   │ AnemoiModelInterface │ 57.4 M │ train │     0 │
│ 1 │ loss    │ ModuleDict           │      0 │ train │     0 │
│ 2 │ metrics │ ModuleDict           │      0 │ train │     0 │
└───┴─────────┴──────────────────────┴────────┴───────┴───────┘
Trainable params: 57.4 M
Non-trainable params: 0
Total params: 57.4 M
Total estimated model params size (MB): 229
Modules in train mode: 302
Modules in eval mode: 0
Total FLOPs: 0
```

## Results

* Obs metrics at 15km over CONUS show the biggest differences.
  * The surface variables (10m winds, 2m temperature) show a consistent boost
    from using the trainable parameters. Even if it is maybe not statistically
    significant, it is consistent across the board for all variables, indicating
    something is important.
    This skill gain is 0-5 days for winds, and really present for all 10 days in
    2m temperature.
    Surface pressure shows some improvement with the parameters after 5 days
  * Pressure level variables show less consistent results. T850 shows some
    improvement (days 3-5) with the parameters, although it's relatively small.
    Even smaller improvements noticeable in the winds at 3-5 days.
    Otherwise it appears to be a toss up.

* Against HRRR analysis:
  * Similar interpretation for surface variables as with the observations: small
    but consistent improvements for surface pressure, 10m winds, and 2m
    temperature
  * Otherwise no real difference

* Obs metrics at 1 degree
  * Southern Hemisphere:
    * Surface pressure shows slight degradation with the parameters on, all
      other surface variables show no noticeable difference.
    * Pressure level variables show no difference.
  * Northern Hemisphere:
    * 10m winds show slight degradation with the parameters after ~7 days,
      whereas very slight improvement in 2m temperature with parameters.
    * Pressure levels largely show no differnce, although maybe a slight
      improvement with the parameters in day 5 T850.
  * Europe:
    * Surface pressure and T850, T250 show slight benefit to the parameters
  * Global:
    * Insignificant degradation in 10m winds beyond day 7, and insignificant
      improvement with the parameters in 2m temperature, and surface pressure
    * Small, insignificant improvement in T850, Z850 with the parameters, otherwise
      it's hard to see any differences.


