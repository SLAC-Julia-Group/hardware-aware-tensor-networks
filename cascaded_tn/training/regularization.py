# Put this in cascaded_tn/training/regularization.py (new file)

import jax
import jax.numpy as jnp
import numpy as np

from tn4ml.metrics import LogReLUFrobNorm

def LogAnomalyRegNorm(model, alpha_spectral=0.2, alpha_smooth=0.1, alpha_sparse=0.3):
    """
    Drop-in replacement for LogReLUFrobNorm with custom anomaly detection regularization.
    
    Works exactly like LogReLUFrobNorm but implements our discussed regularization strategy.
    Can be used directly with CombinedLoss.
    """
    # Get the cascade from the model
    cascade = model.cascade if hasattr(model, 'cascade') else model
    
    if not hasattr(cascade, 'operators'):
        # Fallback to standard norm if not a cascade
        return LogReLUFrobNorm(model)
    
    reg_total = 0.0
    prev_norm = None
    
    for i, op in enumerate(cascade.operators):
        if hasattr(op, 'implementation') and hasattr(op.implementation, 'tensors'):
            tensors = op.implementation.tensors
            
            # 1. Spectral regularization
            for tensor in tensors:
                shape = tensor.shape
                if len(shape) >= 2:
                    matrix = tensor.data.reshape(shape[0], -1)
                    s = jnp.linalg.svd(matrix, compute_uv=False)
                    effective_rank = jnp.sum(s) ** 2 / (jnp.sum(s ** 2) + 1e-8)
                    reg_total += alpha_spectral * (1.0 / (effective_rank + 0.1))
            
            # 2. Layer smoothness
            if i > 0 and prev_norm is not None:
                curr_norm = sum(jnp.sum(t.data**2) for t in tensors)
                reg_total += alpha_smooth * jnp.abs(jnp.log(curr_norm / (prev_norm + 1e-8)))
                prev_norm = curr_norm
            elif prev_norm is None:
                prev_norm = sum(jnp.sum(t.data**2) for t in tensors)
            
            # 3. Bottleneck sparsity (last layer)
            if i == len(cascade.operators) - 1:
                for tensor in tensors:
                    reg_total += alpha_sparse * jnp.sum(jnp.abs(tensor.data))
    
    # Return log of the regularization (matching LogReLUFrobNorm format)
    return jnp.log(jnp.maximum(reg_total, 1e-10))