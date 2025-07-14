# MMGT - TNBD

Multi Mesh Graph Transformer with
* trimmed GFS/HRRR boundary, to 15-16 grid cells (so 75-80km) on boundaries
* No Boundary Disconnected nodes

This is a copy of `../mmgt-nbd` except that it additionally trims the boundaries
of the HRRR dataset.
It turns out that the same number of encoder nearest neighbors (20) is required
to make it so there are no disconnected nodes.
