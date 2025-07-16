# MMGT - NBD

Multi Mesh Graph Transformer with No Boundary Disconnected nodes.
This is a copy of `../mmgt` except that the graph definition was modified so
that there are no disconnected graph nodes around the GFS/HRRR boundary.
Note that there are still disconnected nodes close to the poles, but we ignore
these.

The only real parameter to modify so that we don't just discard hanging nodes is
the number of nearest neighbors used to define the encode/decode graphs.
So, this just has a larger KNN parameter, where this was chosen to approximately
be the smallest without hanging nodes near CONUS.

```
  | Name    | Type                 | Params | Mode
---------------------------------------------------------
0 | model   | AnemoiModelInterface | 73.5 M | train
1 | loss    | WeightedMSELoss      | 0      | train
2 | metrics | ModuleList           | 0      | train
---------------------------------------------------------
73.5 M    Trainable params
0         Non-trainable params
73.5 M    Total params
294.179   Total estimated model params size (MB)
309       Modules in train mode
0         Modules in eval mode
```
