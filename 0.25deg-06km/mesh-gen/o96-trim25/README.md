# Updated Mesh

Some sizes:
* There are 65,346 latent mesh nodes
* About 40% (26,108) are in the nest (whereas with the coarse model 58% were)
* If we consider all points in both subdomains from ~53 degrees north to ~22
  degrees north, there are 32,946 (~50%)

## TODO

* [x] Max cutoff for global is 74, that's good to go.
* [x] Now just need to trim down the max num. neareast neighboUrs
* [ ] Confirm max cutoff = 24km is good in the LAM region
