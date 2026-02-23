# CRPS Experiments

Here's the progression:
1. `gridscale.yaml`: a comparable setup to `../base-0.8.4-mod/no_truncation.yaml`, which applies 4x grid scale noise at each point on the latent mesh
  * Note I never reran this with the update code, the yaml is just here for
    comparison
2. `noise32.yaml` and `noise128.yaml`: These are the first attempts to inject a
   low dimensional, global (shared) noise vector to the model. However this just
   injects noise into the processor layers, so the impact is really minimal.
3. `sharednoise32.yaml`: apply a shared 32 dimensional noise vector into the
   encoder, processor, and decoder via conditional layer normalization.
   The conditional layer norm involves a matrix multiplication to transform the
   low dim vector into the latent space where it's applied.
    * Note that the query key normalization `qk_norm` is turned on for the
      processor, but this used a typical layer norm, not conditional layer norm
4. `mlpnoise32.yaml`: same as sharednoise setup, but send the noise through a
   single layer MLP, mapping it to a 128 dim vector before sending it to the
   conditional layer norm
    * Note that the query key normalization `qk_norm` is turned on for the
      processor, but this used a typical layer norm, not conditional layer norm
5. `mlpqk32.yaml`: use conditional layer norm for the query, key normalization
   during attention, AND turn on qk normalization in the encoder and processor
   (previously this was only turned on for the processor).
6. `noqk32.yaml`: Turn off all qk normalization, but otherwise this is the same
   as mlpnoise setup.
7. `twostep.yaml`: Same as noqk32, except use 2 timesteps as input.
