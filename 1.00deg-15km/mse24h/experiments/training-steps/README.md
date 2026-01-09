# Test Number of Epochs or Training Iterations

This is to enable comparison between the 72h loss examples, in order to see what
the benefit (or penalty via potential over smoothing)  of extending the loss
window from 24h to 72h.
This uses the same number of iterations as for training with a 72h loss.

Same as the "base" fine tuning in the data-years experiment.

Fixing the following parameters (at least):
* Graph Encoder: 12 KNN determines graph encoding
* Graph Decoder: 3 KNN determines graph decoding
* Processor architecture: sliding window transformer
* Custom latent space, essentially inherited (coarsened) from data space
* Processor window size = 4320
* Model channel width = 512 (except the experiments marked with nc1024)
* Training steps = 30k
* LAM loss fraction = 0.1
* "Empirical" loss weights per-variable group, essentially following AIFS
* Trimmed LAM edge = (10, 11, 10, 11) grid cells  (~150km)
* Loss = 24h, ideally with 1440 steps per epoch increment, but this is not
  panning out to be straightforward for some reason
* Use all data during the fine tuning stage.
