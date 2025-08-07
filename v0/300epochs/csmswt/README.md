# Longer training


These setups use the following settings based on experiments with 30k optim
steps:

* LAM Loss = 10%
* Trim HRRR domain by 10 coarsened grid cells (150 km)
* Processor window size = 4320
* 512 channels

Except the 300 epoch (base) case uses LAM Loss of 50% (fired off early)
