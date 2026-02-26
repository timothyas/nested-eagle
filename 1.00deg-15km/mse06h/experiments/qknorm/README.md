# QK Norm Test

This seems like such a detail, but I started investigating this because the
`qk_norm` is turned on for the processor in the ensemble (CRPS) model by default
in anemoi, whereas is turned off everywhere else (including in the deterministic
model).
Since the ensemble experiments are so expensive, and because training for 30k
iterations with an ensemble setup seems to not be enough for the model to
converge (so comparisons are not seeming to be meaningful) it seemed more efficient to
test the impact of this normalization here.

The setups are:
* `none.yaml`: no query/key normalization in any of the transformers
* `pqk.yaml`: qk normalization in the processor, same as what's turned on by
  default for the ensemble model settings
* `all.yaml`: turn on qk normalization for the encoder, processor, and decoder
  transformers

## Results

It makes absolutely no difference in skill when compared to analysis.
There are some differences when looking at obs comparisons, but given how
inconsistent it is (sometimes it's better with, sometimes without) it seems to
not be a true improvement or necessity in the model.

Additionally, for some reason this normalization is computationally expensive.
It adds ~15-20% compute time for this deterministic setup:
training time goes 4hr 20min to 5hr 20min when "all" is turned on, 5hr ~13min
when it's just on in the processor.

For the ensemble setup it adds ~30% more time, going from ~7hr 30min to  ~9hr
40min with any qk normalization turned on.
Note these times are wildly different from the deterministic setup since I don't
shard the model in the ensemble setup.
