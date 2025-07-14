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
