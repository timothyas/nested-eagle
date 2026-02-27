# Updated Mesh

Some sizes:
* There are 65,346 latent mesh nodes
* About 40% (26,108) are in the nest (whereas with the coarse model 58% were)
* If we consider all points in both subdomains from ~53 degrees north to ~22
  degrees north, there are 32,946 (~50%)

Compared to the initial Lat Lon mesh, this is really superior:
* The original one had ~42k isolated nodes, this has none
* This has about 1.66M fewer data -> hidden edges (~40% fewer)
* This has about 24k fewer mesh nodes (~27% fewer)
* It's about 30 MiB smaller, hopefully that's meaningful
