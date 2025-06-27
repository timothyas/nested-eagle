# Training


## Package stack

```
pip install anemoi-datasets==0.5.23 anemoi-graphs==0.5.2 anemoi-models==0.5.0 anemoi-training==0.4.0
pip install 'earthkit-data<0.14.0'

conda uninstall mlflow mlflow-skinny mlflow-ui

pip install mlflow azureml-core azureml-mlflow
```

With the exception that anemoi-core uses
[this feature branch](https://github.com/timothyas/anemoi-core/tree/feature/azure-mlflow)
with Azure ML MLFlow logging capabilities.


## Run it

```
source ~/.azure.sh
conda activate anemoi
srun --jobid $SLURM_JOBID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
```
