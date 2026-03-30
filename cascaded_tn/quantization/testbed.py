"""
Quantization testbed for evaluating model performance with fixed-point arithmetic.

This module provides tools to:
1. Load trained models
2. Quantize to specified bit-widths
3. Evaluate performance on test data
4. Compare float32 vs quantized metrics
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from .fixed_point import (
    FixedPointConfig, quantize_model_weights, print_quantization_stats,
    quantize_array
)
from ..training.cascaded_model import CascadedModel
from tn4ml.embeddings import embed
from tn4ml.eval import get_roc_curve_data
from sklearn.metrics import auc


class QuantizationTestbed:
    """Testbed for evaluating quantized model performance."""
    
    def __init__(self, model: CascadedModel, embedding):
        """
        Initialize testbed with a trained model.
        
        Args:
            model: Trained CascadedModel
            embedding: Embedding function used during training
        """
        self.model_float = model
        self.embedding = embedding
        self.quantized_weights = None
        self.config = None
        self.stats = None
        
    def quantize(self, n_word: int, n_frac: int, signed: bool = True):
        """
        Quantize the model to specified fixed-point precision.
        
        Args:
            n_word: Total bit width
            n_frac: Fractional bits
            signed: Use signed representation
        """
        print(f"\n[QUANTIZATION] Quantizing model to {n_word} bits ({n_word-n_frac-1 if signed else n_word-n_frac} integer, {n_frac} fractional)")
        
        # Create config
        self.config = FixedPointConfig(n_word=n_word, n_frac=n_frac, signed=signed)
        print(f"  Config: {self.config}")
        
        # Get float weights
        float_weights = [np.array(jax.device_get(arr)) for arr in self.model_float.arrays]
        
        # Quantize
        self.quantized_weights, self.stats = quantize_model_weights(float_weights, self.config)
        
        # Print statistics
        print_quantization_stats(self.stats, self.config)

    def quantize_mixed(self, layer_configs: List[Tuple[int, int]], signed: bool = True):
        """
        Quantize each layer to different precision.
        
        Args:
            layer_configs: List of (n_word, n_frac) tuples, one per layer
                          e.g., [(18, 12), (16, 10), (12, 8)]
            signed: Use signed representation
        """
        n_layers = len(self.model_float.cascade.operators)
        
        if len(layer_configs) != n_layers:
            raise ValueError(
                f"Must provide config for each layer. Model has {n_layers} layers, "
                f"got {len(layer_configs)} configs."
            )
        
        print(f"\n[MIXED QUANTIZATION] Quantizing {n_layers} layers with different precisions")
        
        # Store configs and quantized weights per layer
        self.mixed_configs = []
        self.mixed_quantized_weights = []
        self.mixed_stats = []
        
        # Get tensor indices for each layer
        tensor_idx = 0
        for layer_idx, (n_word, n_frac) in enumerate(layer_configs):
            # Create config for this layer
            config = FixedPointConfig(n_word=n_word, n_frac=n_frac, signed=signed)
            self.mixed_configs.append(config)
            
            # Get this layer's tensors
            op = self.model_float.cascade.operators[layer_idx]
            if hasattr(op, 'implementation') and hasattr(op.implementation, 'tensors'):
                n_tensors_in_layer = len(op.implementation.tensors)
            else:
                n_tensors_in_layer = op.config.input_dim  # Fallback estimate
            
            # Get float weights for this layer
            layer_weight_arrays = self.model_float.arrays[tensor_idx:tensor_idx + n_tensors_in_layer]
            layer_weight_arrays = [np.array(jax.device_get(arr)) for arr in layer_weight_arrays]
            
            # Quantize this layer
            quantized, stats = quantize_model_weights(layer_weight_arrays, config)
            self.mixed_quantized_weights.append(quantized)
            self.mixed_stats.append(stats)
            
            print(f"  Layer {layer_idx}: {config} - {stats['total_params']} params, "
                  f"{stats['clipped_params']} clipped")
            
            tensor_idx += n_tensors_in_layer
        
        print(f"✅ Mixed quantization complete")

    def evaluate_float(self, dataloader, metric_fn, verbose: bool = True) -> np.ndarray:
        """
        Evaluate model in float32 precision.
        
        Args:
            dataloader: Data loader
            metric_fn: Function that computes metric (e.g., loss, score)
            verbose: Show progress bar
            
        Returns:
            Array of metric values for each sample
        """
        all_metrics = []
        
        iterator = tqdm(dataloader, desc="Evaluating (float32)") if verbose else dataloader
        
        for batch in iterator:
            batch = jax.numpy.array(batch, dtype=jnp.float32)
            params = self.model_float.arrays
            
            metrics = metric_fn(batch, None, *params)
            metrics = jax.device_get(metrics)
            all_metrics.append(metrics)
        
        return np.concatenate(all_metrics)
    
    def evaluate_quantized(self, dataloader, metric_fn, verbose: bool = True) -> np.ndarray:
        """
        Evaluate model with quantized weights.
        
        Args:
            dataloader: Data loader
            metric_fn: Function that computes metric
            verbose: Show progress bar
            
        Returns:
            Array of metric values for each sample
        """
        if self.quantized_weights is None:
            raise ValueError("Model not quantized! Call quantize() first.")
        
        all_metrics = []
        
        # Convert quantized weights back to float (with quantization error baked in)
        quantized_as_float = [w().astype(np.float32) for w in self.quantized_weights]
        quantized_as_float = [jnp.array(w) for w in quantized_as_float]
        
        iterator = tqdm(dataloader, desc=f"Evaluating ({self.config})") if verbose else dataloader
        
        for batch in iterator:
            # Quantize input data
            batch_np = np.array(batch, dtype=np.float32)
            batch_quantized = quantize_array(batch_np, self.config)
            batch_float = jnp.array(batch_quantized().astype(np.float32))
            
            # Evaluate with quantized weights and quantized inputs
            metrics = metric_fn(batch_float, None, *quantized_as_float)
            metrics = jax.device_get(metrics)
            all_metrics.append(metrics)
        
        return np.concatenate(all_metrics)

    def evaluate_mixed(self, dataloader, metric_fn, verbose: bool = True) -> np.ndarray:
        """
        Evaluate model with mixed-precision layers.
        
        Args:
            dataloader: Data loader
            metric_fn: Function that computes metric
            verbose: Show progress bar
            
        Returns:
            Array of metric values for each sample
        """
        if not hasattr(self, 'mixed_quantized_weights'):
            raise ValueError("Model not quantized with mixed precision! Call quantize_mixed() first.")
        
        all_metrics = []
        
        # Prepare quantized weights - flatten and convert to float
        all_quantized_weights = []
        for layer_quantized in self.mixed_quantized_weights:
            for w in layer_quantized:
                all_quantized_weights.append(w().astype(np.float32))
        all_quantized_weights = [jnp.array(w) for w in all_quantized_weights]
        
        # Create config string for progress bar
        config_str = " → ".join([str(c) for c in self.mixed_configs])
        iterator = tqdm(dataloader, desc=f"Evaluating (mixed: {config_str})") if verbose else dataloader
        
        for batch in iterator:
            # Quantize input data to first layer's precision
            batch_np = np.array(batch, dtype=np.float32)
            batch_quantized = quantize_array(batch_np, self.mixed_configs[0])
            batch_float = jnp.array(batch_quantized().astype(np.float32))
            
            # Evaluate with mixed-precision quantized weights
            metrics = metric_fn(batch_float, None, *all_quantized_weights)
            metrics = jax.device_get(metrics)
            all_metrics.append(metrics)
        
        return np.concatenate(all_metrics)

    def compare_performance(self,
                           background_loader,
                           signal_loaders: Dict[str, any],
                           metric_fn,
                           save_path: Optional[str] = None) -> Dict:
        """
        Compare float32 vs quantized performance on background and signals.
        
        Args:
            background_loader: Background data loader
            signal_loaders: Dict of {'signal_name': loader}
            metric_fn: Metric function (higher = more anomalous)
            save_path: Optional path to save results
            
        Returns:
            Dictionary with comparison results
        """
        if self.quantized_weights is None:
            raise ValueError("Model not quantized! Call quantize() first.")
        
        results = {
            'config': str(self.config),
            'background': {},
            'signals': {}
        }
        
        # Evaluate background
        print("\n" + "="*80)
        print("EVALUATING BACKGROUND")
        print("="*80)
        
        bkg_float = self.evaluate_float(background_loader, metric_fn)
        bkg_quant = self.evaluate_quantized(background_loader, metric_fn)
        
        results['background'] = {
            'float32': bkg_float,
            'quantized': bkg_quant,
            'mean_diff': float(np.mean(bkg_quant - bkg_float)),
            'std_diff': float(np.std(bkg_quant - bkg_float)),
        }
        
        print(f"\nBackground scores:")
        print(f"  Float32:   mean={np.mean(bkg_float):.4f}, std={np.std(bkg_float):.4f}")
        print(f"  Quantized: mean={np.mean(bkg_quant):.4f}, std={np.std(bkg_quant):.4f}")
        print(f"  Difference: {results['background']['mean_diff']:.4f} ± {results['background']['std_diff']:.4f}")
        
        # Evaluate each signal
        for signal_name, signal_loader in signal_loaders.items():
            print("\n" + "="*80)
            print(f"EVALUATING SIGNAL: {signal_name}")
            print("="*80)
            
            sig_float = self.evaluate_float(signal_loader, metric_fn)
            sig_quant = self.evaluate_quantized(signal_loader, metric_fn)
            
            # Compute ROC curves
            fpr_float, tpr_float = get_roc_curve_data(bkg_float, sig_float, anomaly_det=True)
            auc_float = auc(fpr_float, tpr_float)
            
            fpr_quant, tpr_quant = get_roc_curve_data(bkg_quant, sig_quant, anomaly_det=True)
            auc_quant = auc(fpr_quant, tpr_quant)
            
            results['signals'][signal_name] = {
                'float32': sig_float,
                'quantized': sig_quant,
                'auc_float': float(auc_float),
                'auc_quantized': float(auc_quant),
                'auc_degradation': float(auc_float - auc_quant),
                'auc_degradation_pct': float(100 * (auc_float - auc_quant) / auc_float),
            }
            
            print(f"\n{signal_name} performance:")
            print(f"  AUC (float32):   {auc_float:.4f}")
            print(f"  AUC (quantized): {auc_quant:.4f}")
            print(f"  Degradation:     {results['signals'][signal_name]['auc_degradation']:.4f} ({results['signals'][signal_name]['auc_degradation_pct']:.2f}%)")
        
        # Save results if requested
        if save_path:
            np.savez(save_path, **results)
            print(f"\n[SAVED] Results saved to {save_path}")
        
        return results
    
    def plot_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Plot comparison of float32 vs quantized performance."""
        n_signals = len(results['signals'])
        
        fig, axes = plt.subplots(1, n_signals + 1, figsize=(6 * (n_signals + 1), 5))
        if n_signals == 0:
            axes = [axes]
        
        # Plot background distributions
        ax = axes[0]
        bins = np.linspace(
            min(results['background']['float32'].min(), results['background']['quantized'].min()),
            max(results['background']['float32'].max(), results['background']['quantized'].max()),
            50
        )
        ax.hist(results['background']['float32'], bins=bins, alpha=0.5, label='Float32', density=True)
        ax.hist(results['background']['quantized'], bins=bins, alpha=0.5, label=results['config'], density=True)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Background Distribution')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot each signal
        for idx, (signal_name, signal_data) in enumerate(results['signals'].items()):
            ax = axes[idx + 1]
            
            bins_sig = np.linspace(
                min(signal_data['float32'].min(), signal_data['quantized'].min()),
                max(signal_data['float32'].max(), signal_data['quantized'].max()),
                50
            )
            
            ax.hist(signal_data['float32'], bins=bins_sig, alpha=0.5, label=f'Float32 (AUC={signal_data["auc_float"]:.3f})', density=True)
            ax.hist(signal_data['quantized'], bins=bins_sig, alpha=0.5, label=f'{results["config"]} (AUC={signal_data["auc_quantized"]:.3f})', density=True)
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{signal_name}\nΔAUC = {signal_data["auc_degradation"]:.4f} ({signal_data["auc_degradation_pct"]:.1f}%)')
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()