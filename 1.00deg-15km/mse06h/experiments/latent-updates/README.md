# Latent Space Updates

In trying to chase down the band of bad points at 0deg longitude, I realized I
was ordering points incorrectly.
`anemoi.graphs.generate.utils.get_coordinates_ordering`
takes as input, e.g. `coords = np.stack([lat, lon], axis=-1)`, where I was
giving it longitude first, latitude second.

The experiments here take the `nic/sic` experiment, and correct this ordering.
Additionally:
* fixed.yaml: uses the same latent space definition as before, but now chops
  off global data points within 15km of the nest, and mesh points within 30km of
  the nest. This is what I found reasonable while making the 0.25deg-06km latent
  mesh, so we do it here. It should have little to no noticeable impact.
* heal5.yaml: uses a global healpix mesh at ~200km resolution, along with the
  same coarsened, 30km HRRR mesh (LCC projection) for the nest latent mesh
* o48.yaml: uses a global healpix mesh at ~2 degree resolution, along with the
  same coarsened, 30km HRRR mesh (LCC projection) for the nest latent mesh

## Infinite time

This one would take more time to handle in the postprocessing stage, since all
of my scripts expect an x/y coordinate for the LAM region.
I would only pursue this if it really seems necessary
* heal7heal5.yaml: this uses a global healpix mesh at ~200km resolution
  (refinement level=5) and ~50km (refinement level = 7) over CONUS.

Other concerns:
* Do we need to separate the nested latent space in the graph, like in stretched
  tri nodes? or is this only necessary for the multiscale gnn definition
