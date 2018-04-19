# Bilingual-Autoencoder

Code taken from http://sarathchandar.in/crl


# Installation Instructions
The code runs on Python 2.7. Following packages need to be installed:
* Theano
* libgpuarray
* pygpu

## Install conda environment (first time only)
Creating new environment takes some time, since it installs **all** the packages available in the system into the environment. We need to load `python 3.5` since `conda` is available only as part of `python 3.5`. However, we force `conda` to create environment corresponding to `python 2.7`.

```
module load python/3.5
conda create -n <env_name> python=2.7
```
## Install packages (first time only)

```
module load python/3.5
module load cuda
source activate <env_name>
conda env update
```

The last line will update the environment by installing all packages present in `environment.yml`

## Load conda environment (every usage)
```
module load python/3.5
module load cuda
source activate <env_name>
```

For deactivating the current environment, just issue the command `deactivate`, just like `pip`.

## Running on GPU
Load the conda environment as described in previous subsection
```
THEANO_FLAGS='device=cuda,floatX=float32' python -u run.py
```
A sample `run.sub` has been added as well, which can be submitted directly via `qsub run.sub`.
