# Updated Mesh

Some sizes:
* There are 73,841 latent mesh nodes
* About 26% (26,108) are in the nest (whereas with the coarse model 58% were)
* If we consider all points in both subdomains from ~53 degrees north to ~22
  degrees north, there are 35,277 (~48%)

This seems fine to use, but for reducing the number of grid cells, the
octahedral grid is the way to go.
Plus, this needs >1000 data points connected to the mesh points toward the
poles.
It's probably not a huge issue, but it's more than double what's needed for the
octahedral grid.
Seems not ideal to have such a high level of aggregation if it's avoidable.
