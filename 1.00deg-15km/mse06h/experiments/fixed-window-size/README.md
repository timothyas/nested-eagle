# Window Size Experiments, Revisited

These experiments are the same as the ones in `../window-size` except now the
latent space node ordering has been fixed, so the SWIN transformer is actually
grabbing neighboring points along a zonal band, not meridionally.

The experiments here follow the same setup as `../fixed-window-size/heal5.yaml`
with window sizes: (1080, 2160, 3564=`N_latent / 8`, 4320)

## Some notes on mesh size

* There are 28,513 node points in the latent space
* About 58% (16,587) are in the nest
* If we consider all points in both subdomains from ~53 degrees north to ~22
  degrees north (i.e., all latent mesh nodes covered by the processor in the
  zonal bands that align with the HRRR subdomain), there are 18,801 nodes
  (~66%).

## Results

* 1080 is clearly not large enough
* hard to tell clear differences beyond that
    * Against HRRR analysis, 2160 looks large enough for all fields but surface
      pressure, geopotential height. Otherwise it's only slightly
      (insignificantly) worse than the larger windows.
    * Considering LAM observation case: they're all much closer, but the 2160
      case does seem to sit slightly higher than 3564 and 4320, especially for
      geopotential. Otherwise, the 3 curves just flip between which is the best.
    * Considering NH observation case: very hard to tell, maybe 2160 is slightly
      worse for geopotential
    * Considering global observation case: pretty much impossible to declare a
      winner
    * Considering Europe: seems like 2160 is a bit too small for geopotential,
      even though it's close.
* Training times:
    * 1080: 3:30
    * 2160: 3:40
    * 3564: 4:08
    * 4320: 4:23
