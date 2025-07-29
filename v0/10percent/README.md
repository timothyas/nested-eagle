# Initial Model Development


## Custom Single Mesh, Shifted Window Transformer

What has mattered?

1. Needed to sort the latent mesh (graph) nodes, using the same block of code
   that's in anemoi-graphs generate/.
2. In the graph generation, the "sponge" layer helped reduce the number of
   disconnected nodes, and reduced the number of nearest neighbors
    * with 1 and 2, the boundary issues were resolved and the model looked OK
      but had poor RMSE (losing to HRRR) and developed very large scale
      instabilities
3. Making the transformer processor window size bigger brought the RMSE to
   somewhere between MMGT and HRRR for most fields except surface pressure and
   geopotential, and removed the stability issue.
   Note there are ~360 (180 + 166 = 346) longitudinal points in the latent mesh,
   which dictates the window size
    * window size = 3 * 360 = 1080 looks good, not quite as good as MMGT

What did not matter?

1. using the graph post processor to sort by edges made no discernable
   difference in skill or speed
