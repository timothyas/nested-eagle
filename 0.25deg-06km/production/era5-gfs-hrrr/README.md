# Perlmutter installation

```
module load nccl/2.21.5
conda create -n eagle -c conda-forge python=3.12 ufs2arco
conda activate eagle
pip install git+https://github.com/timothyas/xmovie.git@feature/gif-scale
pip install anemoi-datasets anemoi-graphs anemoi-models anemoi-training[azure] anemoi-inference anemoi-utils anemoi-transform
pip install "eagle-tools>=0.17.1"
pip install "torch<2.7" torchvision
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
pip install "mlflow-skinny<3.0"
```

## Assumptions

* We can change the attention window size from one stage of transfer learning to the next
* The learning rates and training steps used here are a good idea

