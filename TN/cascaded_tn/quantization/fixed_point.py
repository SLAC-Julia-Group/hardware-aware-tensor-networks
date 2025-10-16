"""
Fixed-point quantization utilities for model evaluation.

Provides tools to quantize models to fixed-point precision and evaluate
performance degradation compared to floating-point.
"""

import numpy as np
import jax.numpy as jnp
from fxpmath import Fxp
from typing import List, Tuple, Optional
import warnings


class FixedPointConfig:
    """Configuration for fixed-point quantization."""
    
    def __init__(self, n_word: int, n_frac: int, signed: bool = True):
        """
        Initialize fixed-point configuration.
        
        Args:
            n_word: Total bit width (e.g., 16, 18, 32)
            n_frac: Number of fractional bits (e.g., 8, 12)
            signed: Whether to use signed representation (default: True)
            
        The integer bits = n_word - n_frac - (1 if signed else 0)
        
        Examples:
            >>> config = FixedPointConfig(n_word=18, n_frac=12)  # Q6.12 (1 sign + 5 int + 12 frac)
            >>> config = FixedPointConfig(n_word=16, n_frac=8)   # Q8.8 (1 sign + 7 int + 8 frac)
        """
        self.n_word = n_word
        self.n_frac = n_frac
        self.signed = signed
        self.n_int = n_word - n_frac - (1 if signed else 0)
        
        # Calculate range
        if signed:
            self.max_val = 2**(self.n_int) - 2**(-n_frac)
            self.min_val = -2**(self.n_int)
        else:
            self.max_val = 2**(n_word - n_frac) - 2**(-n_frac)
            self.min_val = 0
        
        self.resolution = 2**(-n_frac)
    
    def __repr__(self):
        return f"FixedPointConfig(Q{self.n_int}.{self.n_frac}, {self.n_word} bits total, range=[{self.min_val:.6f}, {self.max_val:.6f}], res={self.resolution:.8f})"
    
    def __str__(self):
        return f"Q{self.n_int}.{self.n_frac}"


def quantize_array(array: np.ndarray, config: FixedPointConfig, 
                   overflow: str = 'saturate', rounding: str = 'trunc') -> Fxp:
    """
    Quantize a numpy array to fixed-point representation.
    
    Args:
        array: Input array (float32/float64)
        config: Fixed-point configuration
        overflow: How to handle overflow ('saturate', 'wrap')
        rounding: Rounding mode ('trunc', 'around', 'floor', 'ceil')
        
    Returns:
        Fxp: Fixed-point representation
    """
    return Fxp(array, 
               signed=config.signed,
               n_word=config.n_word,
               n_frac=config.n_frac,
               overflow=overflow,
               rounding=rounding)


def quantize_model_weights(weight_arrays: List[np.ndarray], 
                           config: FixedPointConfig) -> Tuple[List[Fxp], dict]:
    """
    Quantize all model weights to fixed-point.
    
    Args:
        weight_arrays: List of weight arrays from model
        config: Fixed-point configuration
        
    Returns:
        quantized_weights: List of quantized weight arrays
        stats: Dictionary with quantization statistics
    """
    quantized_weights = []
    stats = {
        'n_tensors': len(weight_arrays),
        'total_params': 0,
        'clipped_params': 0,
        'max_value': -np.inf,
        'min_value': np.inf,
        'per_tensor_stats': []
    }
    
    for i, arr in enumerate(weight_arrays):
        # Check for values outside range
        arr_np = np.array(arr)
        out_of_range = np.sum((arr_np > config.max_val) | (arr_np < config.min_val))
        
        # Quantize
        quantized = quantize_array(arr_np, config)
        quantized_weights.append(quantized)
        
        # Collect stats
        tensor_stats = {
            'shape': arr.shape,
            'n_params': arr.size,
            'clipped': int(out_of_range),
            'original_min': float(np.min(arr_np)),
            'original_max': float(np.max(arr_np)),
            'quantized_min': float(np.min(quantized())),
            'quantized_max': float(np.max(quantized())),
        }
        
        stats['per_tensor_stats'].append(tensor_stats)
        stats['total_params'] += arr.size
        stats['clipped_params'] += out_of_range
        stats['max_value'] = max(stats['max_value'], float(np.max(arr_np)))
        stats['min_value'] = min(stats['min_value'], float(np.min(arr_np)))
    
    return quantized_weights, stats


def fixed_point_forward_pass(quantized_weights: List[Fxp],
                             input_data: np.ndarray,
                             config: FixedPointConfig,
                             model_apply_fn,
                             embedding) -> np.ndarray:
    """
    Perform forward pass with fixed-point arithmetic.
    
    This is a simulation - actual operations use fixed-point precision.
    
    Args:
        quantized_weights: Quantized model weights
        input_data: Input data (will be quantized)
        config: Fixed-point configuration
        model_apply_fn: Model's apply function
        embedding: Embedding function
        
    Returns:
        Output in float format (for evaluation)
    """
    # Note: This is a simplified version. Full implementation would require
    # reimplementing the forward pass with fixed-point operations at each step.
    # For now, we convert back to float for the operations but track precision loss.
    
    # Convert quantized weights back to float (with quantization error)
    float_weights = [w().astype(np.float32) for w in quantized_weights]
    
    # Quantize input
    input_quantized = quantize_array(input_data, config)
    input_float = input_quantized().astype(np.float32)
    
    # TODO: This is where you'd implement the actual fixed-point forward pass
    # For now, we're using float operations on quantized values
    
    return float_weights, input_float


def print_quantization_stats(stats: dict, config: FixedPointConfig):
    """Print quantization statistics."""
    print("\n" + "="*80)
    print(f"QUANTIZATION STATISTICS: {config}")
    print("="*80)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Clipped parameters: {stats['clipped_params']:,} ({100*stats['clipped_params']/stats['total_params']:.2f}%)")
    print(f"Weight range: [{stats['min_value']:.6f}, {stats['max_value']:.6f}]")
    print(f"Quantization range: [{config.min_val:.6f}, {config.max_val:.6f}]")
    print(f"Resolution: {config.resolution:.8f}")
    
    if stats['clipped_params'] > 0:
        warnings.warn(f"{stats['clipped_params']} parameters were clipped during quantization!")
    
    print("\nPer-tensor statistics:")
    for i, tensor_stats in enumerate(stats['per_tensor_stats']):
        if tensor_stats['clipped'] > 0:
            print(f"  Tensor {i}: {tensor_stats['shape']} - {tensor_stats['clipped']} clipped "
                  f"(range: [{tensor_stats['original_min']:.4f}, {tensor_stats['original_max']:.4f}])")