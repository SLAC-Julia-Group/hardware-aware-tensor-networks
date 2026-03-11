# Anomaly Detection on the 40 MHz dataset

# Baseline from: https://arxiv.org/pdf/2006.02516.pdf
# Idea and datasets extracted from https://arxiv.org/pdf/2108.03986

# - 56 inputs (18 objects (4 mu, 4 el, 10 jets) x pt, eta, phi + pt and phi of MET)

# - The rest is just kept standard for now.
# - Embedding: ParticleVectorEmbedding where each qubit represents (pt, eta, phi) of a particle
# - A smooth Huber Loss is used

## ------------------------------------------------------
# imports 
# -------------------------------------------------------
# let's import and set up os env variables
import os
from xml.parsers.expat import errors

os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE") 

#import and configure jax this is needed first for precision?
import jax
jax.config.update("jax_enable_x64", False)
jax.config.update('jax_default_matmul_precision', 'float32')
jax.config.update("jax_platform_name", 'gpu')
print("JAX devices:", jax.devices())
print("JAX device kind:", jax.devices()[0].device_kind)

# configure torch change precision if needed
import torch

#import libraries
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn
from sklearn.metrics import auc
from jax.nn.initializers import *
import matplotlib.pyplot as plt
import json

#import local stuff
from tn4ml.initializers import *
from tn4ml.models.smpo import *
from tn4ml.models.model import *
from tn4ml.embeddings import *
from tn4ml.metrics import *
from tn4ml.util import *
from tn4ml.eval import *

from tn4ml.embeddings import embed, TrigonometricEmbedding
import quimb.tensor as qtn

from cascaded_tn.training.cascaded_model import *
from cascaded_tn.builders.dimension_calculator import DimensionCalculator

import utils.dataloader as d

from utils.training_tools import *
from tqdm import tqdm
import time

## ------------------------------------------------------
# arg parsing for inputs here
# -------------------------------------------------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for anomaly detection on 40 MHz dataset using cascaded TN autoencoder.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output_folder', type=str, default="Huber_4096BatchSize_200epoch_cascade_norelu_delta25", help='Output folder name for results.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training.')

    return parser.parse_args()

def initialize_torch(seed):
    torch.random.manual_seed(seed)
    torch.set_default_dtype(torch.float32)           
    torch.set_float32_matmul_precision("highest")    # avoid TF32 approximations
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("PyTorch CUDA:", torch.cuda.is_available())

target_loss = 50
huber_delta = 25

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

## i am gonna go to main directly, but this block could be optimised quite a bit 
def main():
    args = parse_args()
    seed= args.seed
    initialize_torch(seed)
    outputFolder = args.output_folder
    epochs = args.epochs
    batch_size = args.batch_size
    ## ------------------------------------------------------
    # Load 40 MHz dataset 
    # -------------------------------------------------------
    inputPath = "/global/cfs/cdirs/m2616/sagar/QiML/" #Change this based on wherever your datasets are located.
    print(os.environ["HDF5_USE_FILE_LOCKING"]) #double check this should be false, otherwise there might be issues with reading the hdf5 files.
    skip_MET_eta=False #Set this to True if you want to skip the MET eta feature in the dataset.
    norm=True #Set this to True if you want to normalize the dataset features to zero mean and unit variance.
    cartesian=False #Set this to True if you want to use cartesian coordinates (px, py, pz) instead of (pt, eta, phi).
    normalize_tensors=False #Set to True if you want to normalize each particle tensor to unit norm.
    normalize_MPS=True #Set to True if you want normalize the embedded MPS to unit frob norm.
    ## ------------------------------------------------------
    # configure outputs and logging 
    # -------------------------------------------------------
    # let's set up some utils for output path and storage of results 
    outputPath = "../output/" + outputFolder + "/"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    else:
        print("Output folder already exists. Files might be overwritten.")

    #let's make a dictionary to store the hyperparameters and results
    to_store = {}

    ## ------------------------------------------------------
    # split datasets
    # -------------------------------------------------------
    to_store['batch_size'] = batch_size
    ordering = np.load("optimal_ordering.npy")
    background_train, background_val, background_test = d.create_split_dataloaders(
    file_path=inputPath+"background_for_training.h5",
    batch_size=batch_size,
    split_ratios=(0.70, 0.05, 0.25),
    norm=norm,
    skip_MET_eta=skip_MET_eta,
    cartesian=cartesian,
    shuffle=True,
    shuffle_before_split=False,
    val_batch_size=10000,
    test_batch_size=10000,
    particle_ordering=ordering
)
    # background = d.load_data(inputPath + "background_for_training.h5", batch_size=batch_size, shuffle=True, skip_MET_eta=skip_MET_eta, norm=norm, cartesian=cartesian)

    print("Background shape before:", next(iter(background_train)).shape)

    ## ------------------------------------------------------
    # training setup
    # -------------------------------------------------------
    # define model parameters
    # scale = 2.0
    initializer = jax.nn.initializers.variance_scaling(scale=2.0, mode='fan_avg', distribution='uniform')
    # initializer = gramschmidt('normal', 1e-1)
    key = jax.random.PRNGKey(seed)

    to_store['target_loss'] = target_loss
    # k = 2
    # embedding = PolynomialEmbedding(degree=k, n=1, include_bias=True)
    # embedding = TrigonometricEmbedding(k)
    # phys_dim = (embedding.dim, embedding.dim) # = (2k, 2k) for trigonometric embedding, (k+1, k+1) for polynomial embedding

    embedding = ParticleVectorEmbedding(normalize=normalize_tensors)
    add_identity = True
    boundary='obc' # 'pbc' or 'obc' = periodic or open boundary conditions

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

    #define the model parameters here to store in the hyperparameters file 
    layer_dims=[19,1]
    bond_dims=[4]
    phys_dims=[3,3]
    enable_relu=False
    lr=4e-3
    optimizer=optax.adam
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
    output_positions='center',
    cyclic=cyclic,
    symmetric=symmetric,
    key=key,
    debug=False,
    initializer=initializer,
    add_identity=add_identity,
    boundary=boundary,
    optimizer=optimizer,
    learning_rate=lr,
    loss_function=loss_combined,
    )
    
    print("Number of parameters in the model are: ", autoencoder.nparams())
    to_store['n_params'] = int(autoencoder.nparams())

    result_folder = outputPath + "/plots/"

    os.makedirs(result_folder, exist_ok=True)

    for tensor in autoencoder.tensors:
        print(tensor)

    def loss_fn(data, targets, *params):

        temp_model = autoencoder.copy()

        # Update with new parameters
        for tensor, param in zip(temp_model.tensors, params):
            tensor.modify(data=param)

        # Embed the scalar data to MPS
        tn_i = embed(data, embedding)
    
        return autoencoder.loss(temp_model, tn_i)

    loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, None] + [None]*autoencoder.L), backend=autoencoder.device)

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
            return ChosenLossFunction(temp_model, tn_i)
    
        # Vectorize over batch
        errors = jax.vmap(single_error, in_axes=[0] + [None]*len(params))(sample_batch, *params)
        return jnp.mean(errors)

    grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=range(2, 2+autoencoder.L)), in_axes=[0, None] + [None]*autoencoder.L), backend=autoencoder.device)

    params = autoencoder.arrays
    autoencoder.step, autoencoder.opt_state = autoencoder.create_train_step(params=params, loss_func=loss_func, grads_func=grads_func)
       
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
    ## ------------------------------------------------------
    # training loop 
    # -------------------------------------------------------
    to_store['epochs'] = (epochs)
    history = { 'loss': [], 
                'val_loss' : [],
                'epoch_time': [], 
                'best_epoch': -1,
                'unfinished': True}

    # Initialize early stopping
    early_stopper = SimpleEarlyStopping(patience=50, min_delta=0.0001)

    print(f"[TRAINING] Starting training with early stopping (patience={early_stopper.patience}, min_delta={early_stopper.min_delta})")
    print(f"[DATA] Train batches: {len(background_train)}, Val batches: {len(background_val)}")

    # Create checkpoint directory3
    checkpoint_dir = os.path.join(outputPath, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[CHECKPOINTS] Saving to {checkpoint_dir}")

    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0

        for i, batch in enumerate(tqdm(background_train, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = jax.numpy.array(batch, dtype=jnp.float32)
            params = autoencoder.arrays
            params, autoencoder.opt_state, loss = train_step_jitted(params, autoencoder.opt_state, batch)
            autoencoder.update_tensors(params)
            epoch_loss += loss

        end_time = time.time()
        avg_loss = epoch_loss / len(background_train)

        # Compute individual loss terms for monitoring using the existing loss functions
        # Get current model parameters

        val_loss = evaluate_validation_error(background_val, autoencoder, compute_error_batch)

        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(end_time - start_time)

        # Print epoch summary
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.3f} | Val Loss: {val_loss:.3f} | "
              f"Time: {end_time - start_time:.2f}s")

        # Early stopping check
        if early_stopper.check(val_loss, autoencoder, epoch):
            print(f"[EARLY STOPPING] Training stopped at epoch {epoch + 1}")
            history['best_epoch'] = early_stopper.best_epoch

            # Restore best model
            early_stopper.restore_best_model(autoencoder)

            # Mark as finished (even though stopped early, it's a successful completion)
            history['unfinished'] = False
            break
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pkl")
        autoencoder.save(checkpoint_path)
        print(f"[CHECKPOINT] Saved epoch {epoch+1} to {checkpoint_path}")

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
    print(f"  Final validation error: {history['val_loss'][-1]:.3f}")
    print("="*80)

    # Save final best model
    final_model_path = os.path.join(outputPath, "final_best_model.pkl")
    autoencoder.save(final_model_path)
    print(f"[SAVED] Final best model saved to {final_model_path}")

    ## ------------------------------------------------------
    # let's save the hyperparameters used and the traning history
    # -------------------------------------------------------
    os.makedirs(outputPath, exist_ok=True)
    out_file = os.path.join(outputPath, "hyperparameters.json")
    with open(out_file, "w") as f:
        json.dump(to_store, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.plot(range(len(history['loss'])), history['loss'], linestyle="-.", marker='o', label='Training Loss')
    plt.plot(range(len(history['val_loss'])), history['val_loss'], linestyle="-.", marker='o', label='Validation Loss')
    # plt.plot(range(len(history['spectral_term'])), history['spectral_term'], linestyle="-.", marker='o', label='Spectral Term')
    # plt.plot(range(len(history['smooth_term'])), history['smooth_term'], linestyle="-.", marker='o', label='Smooth Term')
    # plt.plot(range(len(history['sparse_term'])), history['sparse_term'], linestyle="-.", marker='o', label='Sparse Term')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.savefig(result_folder+'training_loss.png', dpi=300)

    ## ------------------------------------------------------
    # let's setup code for evaluation here
    # -------------------------------------------------------
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

        # Return only error term (no regularization)
        return InputsToNorm(temp_model, tn_i)

    # Vectorize it for batch processing
    norm_term = jax.jit(jax.vmap(ComputeNorm, in_axes=[0, None] + [None]*autoencoder.L), backend='gpu')

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

        # Return only error term (no regularization)
        return InputsToScore(temp_model, tn_i)

    # Vectorize it for batch processing
    score_term = jax.jit(jax.vmap(ComputeScore, in_axes=[0, None] + [None]*autoencoder.L), backend='gpu')

    def evaluate_model(model, dataloader, loss_func, verbose=True):
        all_scores = []
    
        for batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
            batch = jax.numpy.array(batch, dtype=jnp.float32)
            params = model.arrays
    
            scores = loss_func(batch, None, *params)
            
            scores = jax.device_get(scores)
            all_scores.append(scores)
    
        return np.concatenate(all_scores)#, total_loss / total_samples
    ## ------------------------------------------------------
    # let's run evaluation on test & signal dataset
    # -------------------------------------------------------
    #load signal dataset
    a4l = d.load_data(inputPath + "Ato4l_lepFilter_13TeV.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian, particle_ordering=ordering)
    lq = d.load_data(inputPath + "leptoquark_LOWMASS_lepFilter_13TeV.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian, particle_ordering=ordering)
    htnu = d.load_data(inputPath + "hChToTauNu_13TeV_PU20.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian, particle_ordering=ordering)
    htt = d.load_data(inputPath + "hToTauTau_13TeV_PU20.h5", batch_size=10000, skip_MET_eta=skip_MET_eta, shuffle=True, norm=norm, cartesian=cartesian, particle_ordering=ordering)

    # Run the evaluation on bkg and signals
    bkg_train_norm = evaluate_model(autoencoder, background_train, norm_term, verbose=True)
    bkg_val_norm = evaluate_model(autoencoder, background_val, norm_term, verbose=True)
    bkg_test_norm = evaluate_model(autoencoder, background_test, norm_term, verbose=True)
    a4l_norm = evaluate_model(autoencoder, a4l, norm_term, verbose=True)
    lq_norm = evaluate_model(autoencoder, lq, norm_term, verbose=True)
    htnu_norm = evaluate_model(autoencoder, htnu, norm_term, verbose=True)
    htt_norm  = evaluate_model(autoencoder, htt, norm_term, verbose=True)

    # Run the evaluation on bkg and signals
    bkg_score = evaluate_model(autoencoder, background_test, score_term, verbose=True)
    a4l_score = evaluate_model(autoencoder, a4l, score_term, verbose=True)
    lq_score = evaluate_model(autoencoder, lq, score_term, verbose=True)
    htnu_score = evaluate_model(autoencoder, htnu, score_term, verbose=True)
    htt_score  = evaluate_model(autoencoder, htt, score_term, verbose=True)

    ## ------------------------------------------------------
    # save to csv files
    # -------------------------------------------------------
    np.savetxt(outputPath+'bkg_train_norm.csv', bkg_train_norm, delimiter=',')
    np.savetxt(outputPath+'bkg_val_norm.csv', bkg_val_norm, delimiter=',')
    np.savetxt(outputPath+'bkg_test_norm.csv', bkg_test_norm, delimiter=',')
    np.savetxt(outputPath+'a4l_norm.csv', a4l_norm, delimiter=',')
    np.savetxt(outputPath+'lq_norm.csv', lq_norm, delimiter=',')
    np.savetxt(outputPath+'htnu_norm.csv', htnu_norm, delimiter=',')
    np.savetxt(outputPath+'htt_norm.csv', htt_norm, delimiter=',') 
    np.savetxt(outputPath+'bkg_score.csv', bkg_score, delimiter=',')
    np.savetxt(outputPath+'a4l_score.csv', a4l_score, delimiter=',')
    np.savetxt(outputPath+'lq_score.csv', lq_score, delimiter=',')
    np.savetxt(outputPath+'htnu_score.csv', htnu_score, delimiter=',')
    np.savetxt(result_folder+'htt_score.csv', htt_score, delimiter=',') 

    ## ------------------------------------------------------
    # let's make and save some plots for quality control
    # -------------------------------------------------------

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

    ## ------------------------------------------------------
    # save some rocs
    # -------------------------------------------------------
    fpr_bkg, tpr_bkg = get_roc_curve_data(bkg_score, bkg_score, anomaly_det=True)
    auc_bkg = auc(fpr_bkg, tpr_bkg)
    # Find the score threshold corresponding to background FPR of 1e-5
    idx_fpr = np.searchsorted(fpr_bkg, 1e-5, side="left")
    if idx_fpr >= len(fpr_bkg):
        idx_fpr = -1
    score_threshold = np.sort(bkg_score)[::-1][idx_fpr]
    print("Anomaly score at FPR=1e-5:", score_threshold)

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

    fpr_lq, tpr_lq = get_roc_curve_data(bkg_score, lq_score, anomaly_det=True)
    auc_lq = auc(fpr_lq, tpr_lq)
    fpr_a4l, tpr_a4l = get_roc_curve_data(bkg_score, a4l_score, anomaly_det=True)
    auc_a4l = auc(fpr_a4l, tpr_a4l)
    fpr_htt, tpr_htt = get_roc_curve_data(bkg_score, htt_score, anomaly_det=True)
    auc_htt = auc(fpr_htt, tpr_htt)
    fpr_htnu, tpr_htnu = get_roc_curve_data(bkg_score, htnu_score, anomaly_det=True)
    auc_htnu = auc(fpr_htnu, tpr_htnu)

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
    for fpr, tpr, color, label in [ (fpr_lq, tpr_lq, "blue", "LQ to b tau"), (fpr_a4l, tpr_a4l, "pink", "A to 4l"), (fpr_htnu, tpr_htnu, "green", "H to tau nu"), (fpr_htt, tpr_htt, "violet", "H to tau tau"), ]:
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

    #### ------------------------------------------------------ compression analysis 
        # Test script to analyze SVD compression behavior on intermediate MPS tensors.
    #This helps determine optimal bond dimensions for the cascade.
    # def analyze_mps_singular_values(mps, site_idx=None):
    #     """
    #     Analyze singular value distribution in an MPS.

    #     Args:
    #         mps: The MPS to analyze
    #         site_idx: Specific site to analyze (None = all sites)

    #     Returns:
    #         Dictionary with singular value information
    #     """
    #     results = {}

    #     if site_idx is not None:
    #         sites = [site_idx]
    #     else:
    #         sites = range(len(mps.tensors) - 1)

    #     for i in sites:
    #         # Get the tensor and reshape for SVD
    #         if i == 0:
    #             # First tensor: (right_bond, phys)
    #             tensor = mps.tensors[i].data
    #             matrix = tensor.reshape(-1, tensor.shape[-1])
    #         elif i == len(mps.tensors) - 1:
    #             # Last tensor: (left_bond, phys)
    #             continue
    #         else:
    #             # Middle tensor: (left_bond, right_bond, phys)
    #             tensor = mps.tensors[i].data
    #             # Reshape to (left_bond * phys, right_bond)
    #             matrix = tensor.reshape(tensor.shape[0] * tensor.shape[-1], tensor.shape[1])

    #         # Compute SVD
    #         U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    #         results[f'site_{i}'] = {
    #             'singular_values': np.array(S),
    #             'bond_dim_original': len(S),
    #             'matrix_shape': matrix.shape,
    #             'cumulative_sum': np.cumsum(S) / np.sum(S),  # Cumulative importance
    #             'effective_rank': np.sum(S)**2 / np.sum(S**2)  # Participation ratio
    #         }

    #     return results


    # def test_compression_levels(model, test_input, embedding, layer_idx=2, bond_dims=[4, 8, 16, 32, 64, 128]):
    #     """
    #     Test different compression levels on a specific layer's output.

    #     Args:
    #         model: Your trained autoencoder
    #         test_input: Single test input
    #         embedding: Embedding to use
    #         layer_idx: Which layer's output to analyze (0, 1, or 2)
    #         bond_dims: List of max bond dimensions to test

    #     Returns:
    #         Dictionary with compression test results
    #     """
    #     print(f"Testing compression on layer {layer_idx} output...")

    #     # Get cascade
    #     cascade = model.cascade

    #     # Convert input
    #     if hasattr(test_input, 'cpu'):
    #         test_input = test_input.cpu().numpy()
    #     test_input = jnp.array(test_input)

    #     # Process through cascade up to specified layer
    #     current_mps = embed(test_input, embedding)

    #     for i in range(layer_idx + 1):
    #         print(f"\nProcessing layer {i}...")
    #         current_mps = cascade.operators[i].apply(current_mps)
    #         print(f"Output MPS: {len(current_mps.tensors)} tensors")

    #         # Check bond dimensions
    #         bonds = []
    #         for j in range(len(current_mps.tensors) - 1):
    #             if j < len(current_mps.tensors[j].shape) - 1:
    #                 bonds.append(current_mps.tensors[j].shape[1] if len(current_mps.tensors[j].shape) > 2 else current_mps.tensors[j].shape[0])
    #         print(f"Bond dimensions: {bonds}")

    #     # Analyze singular values BEFORE compression
    #     print(f"\nAnalyzing singular values after layer {layer_idx}...")
    #     sv_analysis = analyze_mps_singular_values(current_mps)

    #     # Store original for comparison
    #     original_mps = current_mps.copy()
    #     original_norm = original_mps.norm()

    #     # Test different compression levels
    #     compression_results = {
    #         'original_bond_dims': bonds,
    #         'singular_value_analysis': sv_analysis,
    #         'compression_tests': {}
    #     }

    #     for max_bond in bond_dims:
    #         print(f"\nTesting compression with max_bond={max_bond}...")

    #         # Compress
    #         compressed_mps = original_mps.copy()
    #         compressed_mps.compress(max_bond=max_bond, cutoff=1e-12)

    #         # Measure difference
    #         diff_norm = (original_mps - compressed_mps).norm()
    #         relative_error = diff_norm / original_norm

    #         # Get compressed bond dimensions
    #         compressed_bonds = []
    #         for j in range(len(compressed_mps.tensors) - 1):
    #             if j < len(compressed_mps.tensors[j].shape) - 1:
    #                 compressed_bonds.append(compressed_mps.tensors[j].shape[1] if len(compressed_mps.tensors[j].shape) > 2 else compressed_mps.tensors[j].shape[0])

    #         # Contract both to compare final outputs
    #         original_contracted = original_mps.contract(all, optimize='auto-hq')
    #         compressed_contracted = compressed_mps.contract(all, optimize='auto-hq')

    #         output_diff = jnp.linalg.norm(original_contracted.data - compressed_contracted.data)
    #         output_relative_error = output_diff / jnp.linalg.norm(original_contracted.data)

    #         compression_results['compression_tests'][max_bond] = {
    #             'compressed_bonds': compressed_bonds,
    #             'relative_error_mps': float(relative_error),
    #             'relative_error_output': float(output_relative_error),
    #             'compression_ratio': bonds[0] / max_bond if bonds else 1.0
    #         }

    #         print(f"  Compressed bonds: {compressed_bonds}")
    #         print(f"  MPS relative error: {relative_error:.2e}")
    #         print(f"  Output relative error: {output_relative_error:.2e}")

    #     return compression_results, original_mps

    # def plot_singular_value_analysis(results, save_path="singular_values_analysis.png"):
    #     """
    #     Plot singular value decay and cumulative importance.
    #     """
    #     sv_analysis = results['singular_value_analysis']

    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #     fig.suptitle('Singular Value Analysis of Intermediate MPS', fontsize=16)

    #     # Plot 1: Singular value decay for first few sites
    #     ax = axes[0, 0]
    #     for i, (site, data) in enumerate(list(sv_analysis.items())[:3]):
    #         sv = data['singular_values']
    #         ax.semilogy(sv, 'o-', label=f'{site} (dim={len(sv)})', alpha=0.7)
    #     ax.set_xlabel('Index')
    #     ax.set_ylabel('Singular Value')
    #     ax.set_title('Singular Value Decay (log scale)')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)

    #     # Plot 2: Cumulative importance
    #     ax = axes[0, 1]
    #     for i, (site, data) in enumerate(list(sv_analysis.items())[:3]):
    #         cumsum = data['cumulative_sum']
    #         ax.plot(cumsum, '-', label=f'{site}', linewidth=2)
    #         # Mark 99% and 99.9% thresholds
    #         idx_99 = np.argmax(cumsum >= 0.99)
    #         idx_999 = np.argmax(cumsum >= 0.999)
    #         ax.axvline(idx_99, color='gray', linestyle='--', alpha=0.5)
    #         ax.axvline(idx_999, color='gray', linestyle=':', alpha=0.5)
    #     ax.set_xlabel('Number of Singular Values')
    #     ax.set_ylabel('Cumulative Importance')
    #     ax.set_title('Cumulative Singular Value Importance')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)

    #     # Plot 3: Compression error vs bond dimension
    #     ax = axes[1, 0]
    #     compression_tests = results['compression_tests']
    #     bond_dims = sorted(compression_tests.keys())
    #     mps_errors = [compression_tests[d]['relative_error_mps'] for d in bond_dims]
    #     output_errors = [compression_tests[d]['relative_error_output'] for d in bond_dims]

    #     ax.semilogy(bond_dims, mps_errors, 'o-', label='MPS Error', linewidth=2)
    #     ax.semilogy(bond_dims, output_errors, 's-', label='Output Error', linewidth=2)
    #     ax.set_xlabel('Max Bond Dimension')
    #     ax.set_ylabel('Relative Error')
    #     ax.set_title('Compression Error vs Bond Dimension')
    #     ax.legend()
    #     ax.grid(True, alpha=0.3)

    #     # Plot 4: Effective rank distribution
    #     ax = axes[1, 1]
    #     effective_ranks = [data['effective_rank'] for data in sv_analysis.values()]
    #     sites = list(range(len(effective_ranks)))
    #     ax.bar(sites, effective_ranks, alpha=0.7)
    #     ax.set_xlabel('Site Index')
    #     ax.set_ylabel('Effective Rank')
    #     ax.set_title('Effective Rank by Site')
    #     ax.grid(True, alpha=0.3, axis='y')

    #     plt.tight_layout()
    #     plt.savefig(result_folder+save_path, dpi=150, bbox_inches='tight')
    #     plt.show()

    #     # Print summary
    #     print("\n" + "="*60)
    #     print("COMPRESSION ANALYSIS SUMMARY")
    #     print("="*60)

    #     # Find optimal compression
    #     for site, data in list(sv_analysis.items())[:1]:  # Just show first site
    #         sv = data['singular_values']
    #         cumsum = data['cumulative_sum']

    #         print(f"\n{site}:")
    #         print(f"  Original bond dimension: {len(sv)}")
    #         print(f"  Effective rank: {data['effective_rank']:.2f}")
    #         print(f"  SV range: [{sv[0]:.3f}, {sv[-1]:.2e}]")
    #         print(f"  To keep 99% info: {np.argmax(cumsum >= 0.99) + 1} dimensions")
    #         print(f"  To keep 99.9% info: {np.argmax(cumsum >= 0.999) + 1} dimensions")
    #         print(f"  To keep 99.99% info: {np.argmax(cumsum >= 0.9999) + 1} dimensions")

    #     print("\nCompression recommendations:")
    #     for max_bond, data in compression_tests.items():
    #         if data['relative_error_output'] < 1e-6:
    #             print(f"  Max bond = {max_bond}: Output error = {data['relative_error_output']:.2e} ✓")
    #             break


    # # Example usage function
    # def run_compression_test(model, test_event=None, embedding=None):
    #     """
    #     Run the complete compression test on your model.

    #     Args:
    #         model: Your trained CascadedModel
    #         test_event: Test input (if None, uses random)
    #         embedding: Embedding (if None, uses TrigonometricEmbedding)
    #     """
    #     # Setup defaults
    #     if test_event is None:
    #         input_dim = model.cascade.operators[0].config.input_dim
    #         test_event = np.random.randn(input_dim)
    #         print(f"Using random input of dimension {input_dim}")

    #     if embedding is None:
    #         embedding = TrigonometricEmbedding(k=1)

    #     # Test each layer
    #     all_results = {}

    #     for layer_idx in range(1):  # Test layers 0, 1
    #         print(f"\n{'='*60}")
    #         print(f"TESTING LAYER {layer_idx}")
    #         print(f"{'='*60}")

    #         results, mps = test_compression_levels(
    #             model, 
    #             test_event, 
    #             embedding,
    #             layer_idx=layer_idx,
    #             bond_dims=[2, 4, 6, 8]
    #         )

    #         all_results[f'layer_{layer_idx}'] = results

    #         # Plot for this layer
    #         plot_singular_value_analysis(
    #             results, 
    #             save_path=f"compression_analysis_layer_{layer_idx}.png"
    #         )

    #     # Overall recommendation
    #     print("\n" + "="*80)
    #     print("OVERALL RECOMMENDATIONS")
    #     print("="*80)

    #     for layer_idx in range(2):
    #         results = all_results[f'layer_{layer_idx}']
    #         original_bond = results['original_bond_dims'][0] if results['original_bond_dims'] else 'N/A'

    #         # Find minimal bond dim with acceptable error
    #         min_bond = None
    #         for bond_dim in sorted(results['compression_tests'].keys()):
    #             if results['compression_tests'][bond_dim]['relative_error_output'] < 1e-6:
    #                 min_bond = bond_dim
    #                 break
                
    #         print(f"\nLayer {layer_idx}:")
    #         print(f"  Current bond dim: {original_bond}")
    #         print(f"  Recommended max bond: {min_bond or 'No compression needed'}")
    #         if min_bond and original_bond != 'N/A':
    #             print(f"  Compression factor: {original_bond/min_bond:.2f}x")

    #     return all_results

    # test_event = next(iter(background))[0]
    # results = run_compression_test(autoencoder, test_event, embedding)

if __name__ == "__main__":
        import time
        from tqdm import tqdm
        main()
