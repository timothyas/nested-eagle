# MMGT - TNBD

Multi Mesh Graph Transformer with
* trimmed GFS/HRRR boundary, to 15-16 grid cells (so 75-80km) on boundaries
* No Boundary Disconnected nodes

This is a copy of `../mmgt-nbd` except that it additionally trims the boundaries
of the HRRR dataset.
It turns out that the same number of encoder nearest neighbors (20) is required
to make it so there are no disconnected nodes.

```
  | Name    | Type                 | Params | Mode
---------------------------------------------------------
0 | model   | AnemoiModelInterface | 72.0 M | train
1 | loss    | WeightedMSELoss      | 0      | train
2 | metrics | ModuleList           | 0      | train
---------------------------------------------------------
72.0 M    Trainable params
0         Non-trainable params
72.0 M    Total params
287.909   Total estimated model params size (MB)
309       Modules in train mode
0         Modules in eval mode
```
