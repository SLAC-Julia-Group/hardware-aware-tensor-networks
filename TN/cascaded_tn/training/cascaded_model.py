"""
Training integration for cascaded tensor networks.

This module provides Model classes that integrate with tn4ml's training
infrastructure while using our cascaded architecture.
"""

from typing import Optional, List, Dict, Any, Callable
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
from tn4ml.models.model import Model

from ..core.cascade import TensorNetworkCascade
from ..core.base import debug_timer


class CascadedModel(Model):
    """
    tn4ml Model wrapper for TensorNetworkCascade.
    
    Following the SMPO pattern - inherits from Model and manages tensors internally.
    """
    
    def __init__(self, cascade: TensorNetworkCascade):
        """
        Initialize cascaded model from a cascade.
        
        Args:
            cascade: TensorNetworkCascade to wrap for training
        """
        # Initialize Model first (like SMPO does)
        Model.__init__(self)
        
        # Store cascade reference
        self.cascade = cascade
        self.debug = cascade.debug
        self._cascade_apply = cascade.apply
        
        # Extract and setup tensors
        self._setup_from_cascade(cascade)
        
        if self.debug:
            print(f"[MODEL] Created CascadedModel with {self.L} tensors")
    
    def _setup_from_cascade(self, cascade: TensorNetworkCascade):
        """Setup model from cascade (following SMPO pattern)."""
        # Extract all tensors
        all_tensors = []
        tensor_to_layer = {}
        tensor_idx = 0
        
        for layer_idx, op in enumerate(cascade.operators):
            if hasattr(op, 'implementation'):
                impl = op.implementation
                
                # Get tensors from implementation
                layer_tensors = []
                if hasattr(impl, 'tensors'):
                    layer_tensors = list(impl.tensors)
                elif isinstance(impl, qtn.TensorNetwork):
                    layer_tensors = list(impl.tensors)
                
                # Add to collection
                for t in layer_tensors:
                    all_tensors.append(t)
                    tensor_to_layer[tensor_idx] = layer_idx
                    tensor_idx += 1
        
        # Store as attributes (like SMPO stores its tensors)
        self._tensors = all_tensors
        self._L = len(all_tensors)
        self.tensor_to_layer = tensor_to_layer
    
    @property
    def L(self):
        """Number of tensors."""
        return self._L
    
    @property
    def tensors(self):
        """Access to tensors."""
        return self._tensors
    
    @property
    def arrays(self):
        """Return arrays for tn4ml compatibility."""
        return [t.data for t in self._tensors]
    
    def update_tensors(self, arrays):
        """Update tensor data from arrays (used by tn4ml training)."""
        for tensor, array in zip(self._tensors, arrays):
            tensor.modify(data=array)
        
        # Sync to cascade
        self._sync_parameters_to_cascade()
    
    def apply(self, input_data, **kwargs):
        """
        Apply the model to input data.
        
        Uses the cascade's apply method to maintain proper
        layer-by-layer structure.
        """
        # Ensure cascade has current parameters
        self._sync_parameters_to_cascade()
        
        # Use cascade's apply method
        return self._cascade_apply(input_data, **kwargs)
    
    def _sync_parameters_to_cascade(self):
        """Sync current tensor parameters back to cascade operators."""
        tensor_idx = 0
        
        for layer_idx, op in enumerate(self.cascade.operators):
            if hasattr(op, 'implementation'):
                impl = op.implementation
                
                # Update tensors in implementation
                if hasattr(impl, 'tensors'):
                    # Convert to list if it's a tuple
                    if isinstance(impl.tensors, tuple):
                        impl.tensors = list(impl.tensors)
                    
                    for i in range(len(impl.tensors)):
                        if tensor_idx < len(self._tensors):
                            impl.tensors[i] = self._tensors[tensor_idx]
                            tensor_idx += 1
                
                elif isinstance(impl, qtn.TensorNetwork):
                    # TensorNetwork tensors might also be tuple
                    tensor_list = list(impl.tensors)
                    for i in range(len(tensor_list)):
                        if tensor_idx < len(self._tensors):
                            tensor_list[i] = self._tensors[tensor_idx]
                            tensor_idx += 1
                    # Update the tensor network's tensors
                    impl._tensors = tensor_list
    
    def normalize(self, insert=None):
        """Normalize the model."""
        # Calculate norm through cascade
        norm_val = self.cascade.norm() if hasattr(self.cascade, 'norm') else 1.0
        
        if insert is None:
            # Distribute normalization
            for tensor in self._tensors:
                tensor.modify(data=tensor.data / (norm_val ** (1/self.L)))
        else:
            # Normalize at specific tensor
            if insert < len(self._tensors):
                self._tensors[insert].modify(data=self._tensors[insert].data / norm_val)
    
    def copy(self):
        """Copy the model."""
        # Create a simple copy of the cascade
        new_cascade = TensorNetworkCascade(
            [op for op in self.cascade.operators],
            name=self.cascade.name + "_copy",
            debug=self.cascade.debug,
            validate=False
        )
        return CascadedModel(new_cascade)
    
    def get_encoder_decoder_models(self) -> tuple:
        """
        Split into separate encoder and decoder models.
        
        Returns:
            (encoder_model, decoder_model)
        """
        encoder_cascade = self.cascade.encoder
        decoder_cascade = self.cascade.decoder
        
        encoder_model = CascadedModel(encoder_cascade)
        decoder_model = CascadedModel(decoder_cascade)
        
        return encoder_model, decoder_model
    
    def summary(self):
        """Print model summary."""
        print("\n" + "="*60)
        print("CASCADED MODEL SUMMARY")
        print("="*60)
        
        # Architecture summary
        self.cascade.summary()
        
        # Training configuration
        print("\nTraining Configuration:")
        print(f"  Strategy: {self.strategy}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Loss function: {self.loss}")
        print(f"  Training type: {self.train_type}")
        
        # Parameter count
        try:
            total_params = sum(t.data.size for t in self.tensors)
            print(f"\nTotal parameters: {total_params:,}")
            
            # Layer-wise parameter distribution
            print("\nParameters by layer:")
            for layer_idx in range(len(self.cascade.operators)):
                layer_params = sum(
                    self.tensors[t_idx].data.size 
                    for t_idx, l_idx in self.tensor_to_layer.items() 
                    if l_idx == layer_idx
                )
                op = self.cascade.operators[layer_idx]
                print(f"  Layer {layer_idx} ({op}): {layer_params:,} parameters")
        except Exception as e:
            print(f"\nParameter counting error: {e}")
        
        print("="*60 + "\n")


def create_trainable_autoencoder(layer_dims: List[int],
                               cyclic: bool = False,
                               loss_function=None,
                               optimizer=None,
                               learning_rate: float = 0.01,
                               key=None,
                               **kwargs) -> CascadedModel:
    """
    Convenience function to create a trainable autoencoder.
    
    Creates the cascade and wraps it in a CascadedModel ready for training.
    
    Args:
        layer_dims: Architecture dimensions
        cyclic: Whether to use cyclic boundaries
        loss_function: tn4ml loss function (e.g., LogQuadNorm)
        optimizer: Optimizer (e.g., optax.adam)
        learning_rate: Learning rate
        key: JAX random key
        **kwargs: Additional arguments for cascade creation
        
    Returns:
        CascadedModel ready for training
    """
    from ..builders.autoencoder import AutoencoderBuilder
    
    # Extract debug flag if present
    debug = kwargs.pop('debug', True)
    
    # Create cascade
    builder = AutoencoderBuilder(debug=debug)
    cascade = builder.create_autoencoder(
        layer_dims=layer_dims,
        cyclic=cyclic,
        key=key,
        **kwargs  # Pass remaining kwargs
    )
    
    # Wrap in model
    model = CascadedModel(cascade)
    
    # Configure for training if parameters provided
    if loss_function or optimizer:
        import optax
        
        model.configure(
            loss=loss_function,
            optimizer=optimizer or optax.adam,
            learning_rate=learning_rate,
            train_type='unsupervised',  
            strategy='global'
        )
    
    return model