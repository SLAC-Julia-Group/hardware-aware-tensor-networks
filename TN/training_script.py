#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection on the 40 MHz dataset
# 
# Baseline from: https://arxiv.org/pdf/2006.02516.pdf
# Idea and datasets extracted from https://arxiv.org/pdf/2108.03986
# 
# - 56 inputs (18 objects (4 mu, 4 el, 10 jets) x pt, eta, phi + pt and phi of MET)
# 
# - The rest is just kept standard for now.
# - Embedding: PolynomialEmbedding (n=3)
# - Combined loss: $\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \left( \log \left\| P \Phi({X_i}) \right\|_2^2 - 1 \right)^2 + \alpha \cdot \mathrm{ReLU}\left(\log(\|P\|_F^2)\right)$

# **Imports**

# In[327]:


# let's import and set up os env variables
import os
os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE") 



# In[328]:


#import and configure jax this is needed first for precision?
import jax
jax.config.update("jax_enable_x64", False)
jax.config.update('jax_default_matmul_precision', 'float32')
jax.config.update("jax_platform_name", 'gpu')
print("JAX devices:", jax.devices())
print("JAX device kind:", jax.devices()[0].device_kind)



# In[329]:


# configure torch change precision if needed
import torch
torch.random.manual_seed(42)
torch.set_default_dtype(torch.float32)           
torch.set_float32_matmul_precision("highest")    # avoid TF32 approximations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("PyTorch CUDA:", torch.cuda.is_available())


# In[330]:


#import libraries
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn
from sklearn.metrics import auc
from jax.nn.initializers import *
import matplotlib.pyplot as plt
import json


# In[331]:


#import local stuff
from tn4ml.initializers import *
from tn4ml.models.smpo import *
from tn4ml.models.model import *
from tn4ml.embeddings import *
from tn4ml.metrics import *
from tn4ml.util import *
from tn4ml.eval import *

from cascaded_tn.training.cascaded_model import *
from cascaded_tn.builders.dimension_calculator import DimensionCalculator


# **Load 40 MHz dataset**

# In[332]:


import utils.dataloader as d

inputPath = "/lus/eagle/projects/ATLAS_workflow_ALCF/sagar/QiML/" #Change this based on wherever your datasets are located.
print(os.environ["HDF5_USE_FILE_LOCKING"]) #double check this should be false, otherwise there might be issues with reading the hdf5 files.
skip_MET_eta=False #Set this to True if you want to skip the MET eta feature in the dataset.
norm=True #Set this to True if you want to normalize the dataset features to zero mean and unit variance.
cartesian=False #Set this to True if you want to use cartesian coordinates (px, py, pz) instead of (pt, eta, phi).
normalize_tensors=False #Set to True if you want to normalize each particle tensor to unit norm.



# In[ ]:


# let's set up some utils for output path and storage of results 
outputFolder = "Huber_AE_Results_bd8_3_hub25"
outputPath = "../output/" + outputFolder + "/"
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
else:
    print("Output folder already exists. Files might be overwritten.")



# In[334]:


#let's make a dictionary to store the hyperparameters and results
to_store = {}


# In[335]:


batch_size = 10000
to_store['batch_size'] = batch_size
background, background_val, background_test = d.create_split_dataloaders(
    file_path=inputPath+"background_for_training.h5",
    batch_size=batch_size,
    split_ratios=(0.5, 0.05, 0.45),
    norm=norm,
    skip_MET_eta=skip_MET_eta,
    cartesian=cartesian,
    shuffle=True,
    shuffle_before_split=False,
)
# background = d.load_data(inputPath + "background_for_training.h5", batch_size=batch_size, shuffle=True, skip_MET_eta=skip_MET_eta, norm=norm, cartesian=cartesian)


# In[336]:


print("Background shape before:", next(iter(background)).shape)


# **Training setup** &nbsp;
# - Stochastic Gradient Descent

# In[ ]:


# define model parameters
# scale = 2.0
initializer = jax.nn.initializers.variance_scaling(scale=2.0, mode='fan_avg', distribution='uniform')
# initializer = gramschmidt('normal', 1e-1)
key = jax.random.PRNGKey(42)
target_loss = 50

to_store['target_loss'] = target_loss

huber_delta = 25
# k = 2
# embedding = PolynomialEmbedding(degree=k, n=1, include_bias=True)
# embedding = TrigonometricEmbedding(k)
# phys_dim = (embedding.dim, embedding.dim) # = (2k, 2k) for trigonometric embedding, (k+1, k+1) for polynomial embedding

embedding = ParticleVectorEmbedding(normalize=normalize_tensors)
add_identity = True
boundary='obc' # 'pbc' or 'obc' = periodic or open boundary conditions


# In[338]:


def AsymmetricLogNorm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Asymmetric loss combining log behavior for small norms and polynomial for large norms.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    x = TransformedNorm(model, data) # This is ||compressed||^2
    # For x < target_loss: log penalty prevents collapse
    # For x > target_loss: add quadratic growth for strong anomaly separation
    log_term = jax.lax.log(x/target_loss)
    symmetric_term = jax.lax.pow(log_term, 2)
    excess = jax.lax.max(0.0, x - target_loss)  # Only penalize when x > target_loss

    return symmetric_term + jax.lax.pow(excess, 2)


# In[339]:


def SymmetricLogNorm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Symmetric loss creating Gaussian-like distribution in log-space around target.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    x = TransformedNorm(model, data)  # ||compressed||^2

    # Main loss: symmetric in log-space
    collapse_penalty = 1e-5 * jax.lax.pow(jax.lax.log(x/target_loss), 2)
    quadratic = jax.lax.pow(x - target_loss, 2)


    return quadratic + collapse_penalty


# In[340]:


def HuberNorm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Huber loss in log-space for robust symmetric training.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    x = TransformedNorm(model, data)
    error = x - target_loss
    abs_error = jax.lax.abs(error)

    # Huber threshold (tune this!)
    delta = huber_delta 

    # Quadratic for small errors, linear for large
    quadratic = 0.5 * jax.lax.pow(error, 2)
    linear = delta * (abs_error - 0.5 * delta)
    huber = jax.lax.select(abs_error <= delta, quadratic, linear)

    collapse_penalty = 1e-5 * jax.lax.pow(jax.lax.log(x/target_loss), 2)


    return huber + collapse_penalty


# In[341]:


def SmoothHuberNorm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Symmetric smooth Huber loss."""
    x = TransformedNorm(model, data)
    error = x - target_loss

    delta = huber_delta  

    # Smooth Huber
    smooth_huber = delta**2 * (jnp.sqrt(1 + (error/delta)**2) - 1)

    # Very weak collapse prevention (only for x << target)
    collapse_penalty = jax.lax.select( x < 1., jax.lax.pow(jax.lax.log(x/target_loss), 2), 0.)

    return smooth_huber + collapse_penalty


# In[342]:


from cascaded_tn.training.regularization import LogAnomalyRegNorm

ChosenLossFunction = SmoothHuberNorm

to_store['loss_function'] = ChosenLossFunction.__name__

if "HuberNorm" in ChosenLossFunction.__name__:
    to_store['huber_delta'] = huber_delta

alpha = 0.04
def loss_combined(*args, **kwargs):
    error = ChosenLossFunction
    # reg = lambda P: alpha * LogAnomalyRegNorm(
    #     P, 
    #     alpha_spectral=0.01,
    #     alpha_smooth=0.2,
    #     alpha_sparse=0
    # )
    reg = lambda P: 0.0
    return CombinedLoss(*args, **kwargs, error=error, reg=lambda P: alpha*reg(P))


# In[343]:


#define the model parameters here to store in the hyperparameters file 
layer_dims=[19,7,3]
bond_dims=[8,3]
phys_dims=[3,2,3]
enable_relu=[0]
lr=1e-3
cyclic=False
symmetric=False

to_store['layer_dims'] = layer_dims
to_store['bond_dims'] = bond_dims
to_store['phys_dims'] = phys_dims
to_store['enable_relu'] = enable_relu
to_store['cyclic'] = cyclic
to_store['symmetric'] = symmetric
to_store['learning_rate'] = lr

autoencoder = create_trainable_autoencoder(
    layer_dims=layer_dims,
    bond_dims=bond_dims,
    phys_dims=phys_dims,
    enable_relu=enable_relu,
    cyclic=cyclic,
    symmetric=symmetric,
    key=key,
    debug=False,
    initializer=initializer,
    add_identity=add_identity,
    boundary=boundary,
    optimizer=optax.adam,
    learning_rate=lr,
    loss_function=loss_combined,
)
# autoencoder = create_trainable_autoencoder(
#     layer_dims=[19,3],
#     bond_dims=[14],
#     phys_dims=[3,3],
#     # enable_relu=[0,1],
#     cyclic=False,
#     symmetric=False,
#     key=key,
#     debug=False,
#     initializer=initializer,
#     add_identity=add_identity,
#     boundary=boundary,
#     optimizer=optax.adam,
#     learning_rate=1e-3,
#     loss_function=loss_combined,
# )


# In[344]:


print("Number of parameters in the model are: ", autoencoder.nparams())
to_store['n_params'] = int(autoencoder.nparams())

result_folder = outputPath + "/plots/"

os.makedirs(result_folder, exist_ok=True)


# In[345]:


def loss_fn(data, targets, *params):

    temp_model = autoencoder.copy()

    # Update with new parameters
    for tensor, param in zip(temp_model.tensors, params):
        tensor.modify(data=param)

    # Embed the scalar data to MPS
    tn_i = embed(data, embedding)

    # Apply the cascade (compression only for now)
    compressed = temp_model.cascade.apply(tn_i)

    return autoencoder.loss(compressed, tn_i)

loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, None] + [None]*autoencoder.L), backend=autoencoder.device)


# In[346]:


# Split into two parts: data-dependent and model-dependent

@jax.jit
def compute_error_batch(sample_batch, *params):
    """JIT-compiled error computation for the batch"""
    def single_error(data_point, *p):
        # Create temp model and update params
        temp_model = autoencoder.copy()
        for tensor, param in zip(temp_model.tensors, p):
            tensor.modify(data=param)

        # Compute error for single point
        tn_i = embed(data_point, embedding)
        compressed = temp_model.cascade.apply(tn_i)
        return ChosenLossFunction(compressed, tn_i)

    # Vectorize over batch
    errors = jax.vmap(single_error, in_axes=[0] + [None]*len(params))(sample_batch, *params)
    return jnp.mean(errors)

# Separate function for regularization (computed ONCE, not per batch item!)
def compute_reg_components_fast(*params):
    """Compute regularization terms - only needs to run once per parameter update"""
    # Pre-compute tensor shapes and indices outside JIT
    tensor_shapes = [t.shape for t in autoencoder.tensors]
    layer_boundaries = []  # Store which tensors belong to which layer

    # Figure out tensor-to-layer mapping
    idx = 0
    for op in autoencoder.cascade.operators:
        if hasattr(op.implementation, 'tensors'):
            n_tensors = len(op.implementation.tensors)
            layer_boundaries.append((idx, idx + n_tensors))
            idx += n_tensors

    @jax.jit
    def reg_terms(params_concat):
        """JIT-compiled regularization computation"""
        # Split concatenated params back into individual tensors
        params_list = []
        start = 0
        for shape in tensor_shapes:
            size = np.prod(shape)
            param = params_concat[start:start+size].reshape(shape)
            params_list.append(param)
            start += size

        reg_spectral = reg_smooth = reg_sparse = 0.0
        prev_norm = None

        # Process each layer
        for layer_idx, (start_idx, end_idx) in enumerate(layer_boundaries):
            layer_tensors = params_list[start_idx:end_idx]

            # 1. Spectral regularization
            for tensor in layer_tensors:
                if len(tensor.shape) >= 2:
                    matrix = tensor.reshape(tensor.shape[0], -1)
                    s = jnp.linalg.svd(matrix, compute_uv=False)
                    effective_rank = jnp.sum(s) ** 2 / (jnp.sum(s ** 2))
                    reg_spectral += (1.0 / (effective_rank + 0.1))

            # 2. Layer smoothness
            curr_norm = sum(jnp.sum(t**2) for t in layer_tensors)
            if layer_idx > 0 and prev_norm is not None:
                reg_smooth += jnp.abs(jnp.log(curr_norm / (prev_norm)))
            prev_norm = curr_norm

            # 3. Bottleneck sparsity (last layer)
            if layer_idx == len(layer_boundaries) - 1:
                for tensor in layer_tensors:
                    reg_sparse += jnp.sum(jnp.abs(tensor))

        return reg_spectral, reg_smooth, reg_sparse

    # Concatenate all parameters into single array
    params_concat = jnp.concatenate([p.flatten() for p in params])

    return reg_terms(params_concat)


# In[347]:


grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=range(2, 2+autoencoder.L)), in_axes=[0, None] + [None]*autoencoder.L), backend=autoencoder.device)

params = autoencoder.arrays
autoencoder.step, autoencoder.opt_state = autoencoder.create_train_step(params=params, loss_func=loss_func, grads_func=grads_func)


# In[348]:


@jax.jit
def train_step_jitted(params, opt_state, batch):
    # Compute loss and gradients in one go
    loss_val, grads = jax.value_and_grad(
        lambda *p: jnp.mean(jax.vmap(loss_fn, in_axes=[0, None] + [None]*len(p))(batch, None, *p)),
        argnums=range(len(params))
    )(*params)

    # Update parameters
    grads_dict = {i: g for i, g in enumerate(grads)}
    params_dict = {i: p for i, p in enumerate(params)}

    updates, opt_state = autoencoder.optimizer.update(grads_dict, opt_state)
    params_dict = optax.apply_updates(params_dict, updates)

    return tuple(params_dict.values()), opt_state, loss_val


# In[349]:


from utils.training_tools import *
from tqdm import tqdm
import time

epochs = 200
to_store['epochs'] = (epochs)
history = { 'loss': [], 
            'val_error' : [],
            'spectral_term': [],
            'smooth_term': [],
            'sparse_term': [],
            'epoch_time': [], 
            'best_epoch': -1,
            'unfinished': True}

# Initialize early stopping
early_stopper = SimpleEarlyStopping(patience=10, min_delta=0.001)

print(f"[TRAINING] Starting training with early stopping (patience={early_stopper.patience}, min_delta={early_stopper.min_delta})")
print(f"[DATA] Train batches: {len(background)}, Val batches: {len(background_val)}")

for epoch in range(epochs):
    start_time = time.time()
    epoch_loss = 0

    for i, batch in enumerate(tqdm(background, desc=f"Epoch {epoch+1}/{epochs}")):
        batch = jax.numpy.array(batch, dtype=jnp.float32)
        params = autoencoder.arrays
        params, autoencoder.opt_state, loss = train_step_jitted(params, autoencoder.opt_state, batch)
        autoencoder.update_tensors(params)
        epoch_loss += loss
        # print(f"Batch {i+1}/{len(background)}: Loss = {loss:.3f}")
        # if i == 0:
        #     break # For debugging, only train on the first batch

    end_time = time.time()
    avg_loss = epoch_loss / len(background)

    # Compute individual loss terms for monitoring using the existing loss functions
    # Get current model parameters

    val_error = evaluate_validation_error(background_val, autoencoder, compute_error_batch)

    reg_spectral, reg_smooth, reg_sparse = compute_reg_components_fast(*autoencoder.arrays)
    reg_spectral = float(reg_spectral)
    reg_smooth = float(reg_smooth)
    reg_sparse = float(reg_sparse)

    history['loss'].append(avg_loss)
    history['val_error'].append(val_error)
    history['spectral_term'].append(reg_spectral)
    history['smooth_term'].append(reg_smooth)
    history['sparse_term'].append(reg_sparse)
    history['epoch_time'].append(end_time - start_time)

    # Print epoch summary
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.3f} | Val Error: {val_error:.3f} | "
          f"Spectral: {reg_spectral:.3f} | Smooth: {reg_smooth:.3f} | Sparse: {reg_sparse:.3f} | "
          f"Time: {end_time - start_time:.2f}s")

    # Early stopping check
    if early_stopper.check(val_error, autoencoder, epoch):
        print(f"[EARLY STOPPING] Training stopped at epoch {epoch + 1}")
        history['best_epoch'] = early_stopper.best_epoch

        # Restore best model
        early_stopper.restore_best_model(autoencoder)

        # Mark as finished (even though stopped early, it's a successful completion)
        history['unfinished'] = False
        break

    # If training completes all epochs
    if epoch == epochs - 1:
        history['unfinished'] = False
        history['best_epoch'] = early_stopper.best_epoch

# Final summary
print("\n" + "="*80)
print("[TRAINING COMPLETE]")
print(f"  Total epochs run: {len(history['loss'])}")
print(f"  Best epoch: {history['best_epoch'] + 1}")
print(f"  Best validation error: {early_stopper.best_loss:.3f}")
print(f"  Final validation error: {history['val_error'][-1]:.3f}")
print("="*80)

# can save training history and model eventually here but skipping for now


# In[350]:


os.makedirs(outputPath, exist_ok=True)
out_file = os.path.join(outputPath, "hyperparameters.json")
with open(out_file, "w") as f:
    json.dump(to_store, f, indent=2)


# In[351]:


plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.plot(range(len(history['loss'])), history['loss'], linestyle="-.", marker='o', label='Training Loss')
plt.plot(range(len(history['val_error'])), history['val_error'], linestyle="-.", marker='o', label='Validation Loss')
# plt.plot(range(len(history['spectral_term'])), history['spectral_term'], linestyle="-.", marker='o', label='Spectral Term')
# plt.plot(range(len(history['smooth_term'])), history['smooth_term'], linestyle="-.", marker='o', label='Smooth Term')
# plt.plot(range(len(history['sparse_term'])), history['sparse_term'], linestyle="-.", marker='o', label='Sparse Term')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.savefig(result_folder+'training_loss.png', dpi=300)


# **Evaluate**

# In[352]:


a4l = d.load_data(inputPath + "Ato4l_lepFilter_13TeV.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian)
lq = d.load_data(inputPath + "leptoquark_LOWMASS_lepFilter_13TeV.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian)
htnu = d.load_data(inputPath + "hChToTauNu_13TeV_PU20.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian)
htt = d.load_data(inputPath + "hToTauTau_13TeV_PU20.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian)


# In[353]:


def InputsToNorm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """The norm itself

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    x = TransformedNorm(model, data)
    return x

@jax.jit
def ComputeNorm(data, targets, *params):
    """Compute only the transformed norm"""
    # Create temp model with updated parameters
    temp_model = autoencoder.copy()
    for tensor, param in zip(temp_model.tensors, params):
        tensor.modify(data=param)

    # Embed the scalar data to MPS
    tn_i = embed(data, embedding)

    # Apply the cascade
    compressed = temp_model.cascade.apply(tn_i)

    # Return only error term (no regularization)
    return InputsToNorm(compressed, tn_i)

# Vectorize it for batch processing
norm_term = jax.jit(jax.vmap(ComputeNorm, in_axes=[0, None] + [None]*autoencoder.L), backend='gpu')


# In[354]:


def InputsToScore(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """The anomaly score based on deviation from target norm.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    x = TransformedNorm(model, data)
    return jax.lax.abs(x - target_loss)

@jax.jit
def ComputeScore(data, targets, *params):
    """Compute the anomaly score"""
    # Create temp model with updated parameters
    temp_model = autoencoder.copy()
    for tensor, param in zip(temp_model.tensors, params):
        tensor.modify(data=param)

    # Embed the scalar data to MPS
    tn_i = embed(data, embedding)

    # Apply the cascade
    compressed = temp_model.cascade.apply(tn_i)

    # Return only error term (no regularization)
    return InputsToScore(compressed, tn_i)

# Vectorize it for batch processing
score_term = jax.jit(jax.vmap(ComputeScore, in_axes=[0, None] + [None]*autoencoder.L), backend='gpu')


# In[355]:


def evaluate_model(model, dataloader, loss_func, embedding, verbose=True):
    all_scores = []

    for batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
        batch = jax.numpy.array(batch, dtype=jnp.float32)
        params = model.arrays

        scores = loss_func(batch, None, *params)

        scores = jax.device_get(scores)
        all_scores.append(scores)

    return np.concatenate(all_scores)#, total_loss / total_samples


# In[356]:


# Run the evaluation on bkg and signals
bkg_train_norm = evaluate_model(autoencoder, background, norm_term, embedding, verbose=True)
bkg_val_norm = evaluate_model(autoencoder, background_val, norm_term, embedding, verbose=True)
bkg_test_norm = evaluate_model(autoencoder, background_test, norm_term, embedding, verbose=True)
a4l_norm = evaluate_model(autoencoder, a4l, norm_term, embedding, verbose=True)
lq_norm = evaluate_model(autoencoder, lq, norm_term, embedding, verbose=True)
htnu_norm = evaluate_model(autoencoder, htnu, norm_term, embedding, verbose=True)
htt_norm  = evaluate_model(autoencoder, htt, norm_term, embedding, verbose=True)


# In[357]:


# Run the evaluation on bkg and signals
bkg_score = evaluate_model(autoencoder, background, score_term, embedding, verbose=True)
a4l_score = evaluate_model(autoencoder, a4l, score_term, embedding, verbose=True)
lq_score = evaluate_model(autoencoder, lq, score_term, embedding, verbose=True)
htnu_score = evaluate_model(autoencoder, htnu, score_term, embedding, verbose=True)
htt_score  = evaluate_model(autoencoder, htt, score_term, embedding, verbose=True)


# **Plot anomaly scores and ROC curve**

# In[358]:


# Train, test, validation norm compatibility plots
plt.figure(figsize=(8, 6))
plt.yscale('log')
xmin = 0
xmax = 150
bins = np.linspace(xmin, xmax, xmax-xmin)
plt.hist(bkg_train_norm, bins=bins, density=True, label='Bkg Train', color='blue', histtype='step')
plt.hist(bkg_val_norm, bins=bins, density=True, label='Bkg Val', color='orange', histtype='step')
plt.hist(bkg_test_norm, bins=bins, density=True, label='Bkg Test', color='green', histtype='step')
plt.axvline(target_loss, color='red', linestyle='--', label='Target Norm')
if "HuberNorm" in ChosenLossFunction.__name__:
   plt.axvline(target_loss - huber_delta, color='purple', linestyle=':', label='Huber Delta')
   plt.axvline(target_loss + huber_delta, color='purple', linestyle=':')
plt.title('Background Norm Distribution')
plt.xlabel('Norm')
plt.ylabel('Density')
plt.xlim([xmin, xmax])
plt.ylim([1e-7, 2])
plt.legend()
plt.savefig(result_folder+'bkg_norm_compatibility.png', dpi=300)


# In[359]:


plt.figure()
plt.yscale('log')
# xmin = 1e-3
# xmax = 1e3
# bins = np.logspace(np.log(xmin), np.log(xmax), 1000)
xmin = 0
xmax = 150
bins = np.linspace(xmin-0.5, xmax-0.5, xmax-xmin)
plt.hist(bkg_test_norm, bins=bins, histtype='step', color='black', label='background', density=True)
plt.hist(lq_norm, bins=bins, histtype='step', color='blue', label='LQ to b tau', density=True)
plt.hist(a4l_norm, bins=bins, histtype='step', color='pink', label='A to 4l', density=True)
plt.hist(htnu_norm, bins=bins, histtype='step', color='green', label='H to tau nu', density=True)
plt.hist(htt_norm, bins=bins, histtype='step', color='violet', label='H to tau tau', density=True)
plt.axvline(target_loss, color='red', linestyle='--', label='Target Norm')
if "HuberNorm" in ChosenLossFunction.__name__:
   plt.axvline(target_loss - huber_delta, color='purple', linestyle=':', label='Huber Delta')
   plt.axvline(target_loss + huber_delta, color='purple', linestyle=':')
plt.title('Norm distributions')
plt.xlim([xmin, xmax])
plt.ylim([1e-7, 2])
# plt.xticks(np.arange(xmin, xmax+1, 1.0))
plt.xlabel('Transformed MPS Norm')
plt.ylabel('Density')
plt.legend()
plt.savefig(result_folder+'transformed_norms.png', dpi=300)


# In[360]:


fpr_bkg, tpr_bkg = get_roc_curve_data(bkg_score, bkg_score, anomaly_det=True)
auc_bkg = auc(fpr_bkg, tpr_bkg)
# Find the score threshold corresponding to background FPR of 1e-5
idx_fpr = np.searchsorted(fpr_bkg, 1e-5, side="left")
if idx_fpr >= len(fpr_bkg):
   idx_fpr = -1
score_threshold = np.sort(bkg_score)[::-1][idx_fpr]
print("Anomaly score at FPR=1e-5:", score_threshold)


# In[361]:


plt.figure()
plt.yscale('log')
# plt.xscale('log')
# xmin = 1e-3
# xmax = 1e3
# bins = np.logspace(np.log(xmin), np.log(xmax), 1000)
xmin = 0
xmax = 150
bins = np.linspace(xmin, xmax, xmax-xmin)
plt.hist(bkg_score, bins=bins, histtype='step', color='black', label='background', density=True)
plt.hist(lq_score, bins=bins, histtype='step', color='blue', label='LQ to b tau', density=True)
plt.hist(a4l_score, bins=bins, histtype='step', color='pink', label='A to 4l', density=True)
plt.hist(htnu_score, bins=bins, histtype='step', color='green', label='H to tau nu', density=True)
plt.hist(htt_score, bins=bins, histtype='step', color='violet', label='H to tau tau', density=True)
plt.axvline(score_threshold, color='red', linestyle='--', label='Score at Bkg FPR=1e-5')
if "HuberNorm" in ChosenLossFunction.__name__:
   plt.axvline(huber_delta, color='purple', linestyle=':', label='Huber Delta')
plt.title('Anomaly score distributions')
plt.xlim([xmin, xmax])
# plt.xticks(np.arange(xmin, xmax+1, 1.0))
plt.xlabel('Anomaly score')
plt.ylabel('Density')
plt.legend()
plt.savefig(result_folder+'anomaly_scores.png', dpi=300)


# In[362]:


# Compute acceptance curves for each signal as a function of anomaly score threshold
thresholds = np.linspace(np.min(bkg_score), np.max(bkg_score), 1000)

def acceptance_curve(scores, thresholds):
   return np.array([(scores >= t).mean() for t in thresholds])

accept_lq = acceptance_curve(lq_score, thresholds)
accept_a4l = acceptance_curve(a4l_score, thresholds)
accept_htt = acceptance_curve(htt_score, thresholds)
accept_htnu = acceptance_curve(htnu_score, thresholds)

plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.plot(thresholds, accept_lq * 100, color="blue", label="LQ to b tau")
plt.plot(thresholds, accept_a4l * 100, color="pink", label="A to 4l")
plt.plot(thresholds, accept_htnu * 100, color="green", label="H to tau nu")
plt.plot(thresholds, accept_htt * 100, color="violet", label="H to tau tau")
# Add horizontal lines and text for acceptance at score_threshold for each signal
for accept, label, color in [
   (accept_lq, "LQ to b tau", "blue"),
   (accept_a4l, "A to 4l", "pink"),
   (accept_htnu, "H to tau nu", "green"),
   (accept_htt, "H to tau tau", "violet"),
]:
   idx = np.searchsorted(thresholds, score_threshold, side="left")
   if idx >= len(accept):
      idx = -1
   acc_val = accept[idx] * 100
   plt.axhline(y=acc_val, xmin=0, xmax=1, color=color, linestyle=":", alpha=0.7)
   plt.text(thresholds[0], acc_val, f"{acc_val:.2f}%", color=color, va="bottom", fontsize=9)
plt.axvline(score_threshold, color='red', linestyle='--', label='FPR=1e-5 threshold')
plt.xlabel("Anomaly Score Threshold")
plt.ylabel("Signal Acceptance (%)")
plt.title("Signal Acceptance vs Anomaly Score Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(result_folder+'signal_acceptance.png', dpi=300)


# In[363]:


fpr_lq, tpr_lq = get_roc_curve_data(bkg_score, lq_score, anomaly_det=True)
auc_lq = auc(fpr_lq, tpr_lq)
fpr_a4l, tpr_a4l = get_roc_curve_data(bkg_score, a4l_score, anomaly_det=True)
auc_a4l = auc(fpr_a4l, tpr_a4l)
fpr_htt, tpr_htt = get_roc_curve_data(bkg_score, htt_score, anomaly_det=True)
auc_htt = auc(fpr_htt, tpr_htt)
fpr_htnu, tpr_htnu = get_roc_curve_data(bkg_score, htnu_score, anomaly_det=True)
auc_htnu = auc(fpr_htnu, tpr_htnu)


# In[364]:


# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-6, 1.0])
plt.ylim([1e-6, 1.0])
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
plt.axvline(1e-5, color="orange", linestyle="-.")
plt.plot(fpr_lq, tpr_lq, color="blue", lw=2, label=f"LQ to b tau (AUC = {auc_lq:.2f})")
plt.plot(fpr_a4l, tpr_a4l, color="pink", lw=2, label=f"A to 4l (AUC = {auc_a4l:.2f})")
plt.plot(fpr_htnu, tpr_htnu, color="green", lw=2, label=f"H to tau nu (AUC = {auc_htnu:.2f})")
plt.plot(fpr_htt, tpr_htt, color="violet", lw=2, label=f"H to tau tau (AUC = {auc_htt:.2f})")

# Add horizontal lines and text for TPR at FPR=1e-5 for each signal
for fpr, tpr, color, label in [
   (fpr_lq, tpr_lq, "blue", "LQ to b tau"),
   (fpr_a4l, tpr_a4l, "pink", "A to 4l"),
   (fpr_htnu, tpr_htnu, "green", "H to tau nu"),
   (fpr_htt, tpr_htt, "violet", "H to tau tau"),
]:
   # Find the TPR at FPR closest to 1e-5
   idx = np.searchsorted(fpr, 1e-5, side="right")
   if idx >= len(tpr):
      idx = -1
   tpr_val = tpr[idx]
   plt.axhline(y=tpr_val, xmin=0, xmax=1, color=color, linestyle=":", alpha=0.7)
   plt.text(1e-1, tpr_val, f"{tpr_val*100:.2f}%", color=color, va="bottom", fontsize=9)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(result_folder+'roc_curves.png', dpi=300)
