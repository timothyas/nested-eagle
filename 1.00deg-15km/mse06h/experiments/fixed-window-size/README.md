# Window Size Experiments, Revisited

These experiments are the same as the ones in `../window-size` except now the
latent space node ordering has been fixed, so the SWIN transformer is actually
grabbing neighboring points along a zonal band, not meridionally.

The experiments here follow the same setup as `../fixed-window-size/heal5.yaml`
with window sizes: (1080, 2160, 3564=N_latent / 8, 4320)

## Some notes on mesh size

* There are 28,513 node points in the latent space
* About 58% (16,587) are in the nest
* If we consider all points in both subdomains from ~53 degrees north to ~22
  degrees north (i.e., all latent mesh nodes covered by the processor in the
  zonal bands that align with the HRRR subdomain), there are 18,801 nodes
  (~66%).
