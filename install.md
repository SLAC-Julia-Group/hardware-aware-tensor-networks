# Installation steps with conda environment


### 1) Create conda environment

```
conda create -n "QiML" python=3.12
conda activate QiML
```

### 2) Check for your gpu specifications

```
nvidia-smi
```

In my case, CUDA Version: 12.2

This means I must install a cudatoolkit with version 12.2 or OLDER (lower number)

### 3) Install cuda toolkit

First check if it's already installed

```
conda list | grep cuda
```

Then check available versions at

https://anaconda.org/nvidia/cuda-toolkit

In my case I choose a  cuda toolkit version <= to that of my driver.

```
conda install -c nvidia/label/cuda-12.2.0 cuda-toolkit=12.2   
```

Then check your installation

```
conda list | grep cuda
nvcc --version
```
### 4) OPTIONAL: install torch

It´s important to do this before installing jax..
```
pip install torch
```


### 5) install jax 

pip is recommended because it updates jax fasater than conda
```
pip install --upgrade pip
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You can validate your installation by running in python

```
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAXlib version:", jax.lib.__version__)
print("JAX backend:", jax.devices())
```

### 6) install remaining libraries

I recommend here that you move to https://github.com/bsc-quantic/tn4ml and continue the installation from there

```
git clone https://github.com/bsc-quantic/tn4ml.git
pip install -e tn4ml/
pip install "tn4ml[examples]"
```

Other optional libraries are

```
pip install jupyterlab notebook
pip install wandb 
```



