"""
Training integration for cascaded tensor networks.

This module provides Model classes that integrate with tn4ml's training
infrastructure while using our cascaded architecture.
"""

from typing import Optional, List, Union, Dict, Any, Callable
import numpy as np
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
        
        # No need to sync - tensors are shared references
        # The modification above already updates the tensors in the cascade
        if self.debug:
            print(f"[MODEL] Updated {len(arrays)} tensor arrays")
    
    def apply(self, input_data, **kwargs):
        """
        Apply the model to input data.
        
        Uses the cascade's apply method to maintain proper
        layer-by-layer structure.
        """
        # Tensors are already updated via shared references
        # Just use cascade's apply method
        return self._cascade_apply(input_data, **kwargs)
    
    def normalize(self, insert=None):
        """Normalize the model."""
        # Calculate norm through cascade
        norm_val = 1.0
        
        # Try to get norm from tensors
        if hasattr(self._tensors[0], 'norm'):
            total_norm = sum(t.norm() ** 2 for t in self._tensors)
            norm_val = jnp.sqrt(total_norm)
        
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
        # Create a deep copy of the cascade
        # This is a simplified version - proper implementation would deep copy tensors
        import copy
        new_cascade = copy.deepcopy(self.cascade)
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
        print(f"  Strategy: {getattr(self, 'strategy', 'Not set')}")
        print(f"  Optimizer: {getattr(self, 'optimizer', 'Not set')}")
        print(f"  Learning rate: {getattr(self, 'learning_rate', 'Not set')}")
        print(f"  Loss function: {getattr(self, 'loss', 'Not set')}")
        print(f"  Training type: {getattr(self, 'train_type', 'Not set')}")
        
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
                               enable_relu: Optional[Union[bool, List[int]]] = None,
                               loss_function=None,
                               optimizer=None,
                               learning_rate: float = 0.01,
                               key=None,
                               phys_dims: Optional[List[int]] = None,
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
        phys_dims: List of physical dimensions [m0, m1, m2, ...] for sequential matching
        **kwargs: Additional arguments for cascade creation
        
    Returns:
        CascadedModel ready for training
    """
    from ..builders.autoencoder import AutoencoderBuilder
    
    # Extract debug flag if present
    debug = kwargs.pop('debug', False)
    
    # Pass phys_dims to the builder
    if phys_dims is not None:
        kwargs['phys_dims'] = phys_dims
    
    # Create cascade
    builder = AutoencoderBuilder(debug=debug)
    cascade = builder.create_autoencoder(
        layer_dims=layer_dims,
        cyclic=cyclic,
        key=key,
        enable_relu=enable_relu,
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
            train_type=0, # 0 = unsupervised training  
            strategy='global',
            device='gpu' if "NVIDIA" in jax.devices()[0].device_kind else 'cpu'
        )
    
    return model


def visualize_cascade_structure(model, show_bonds=True, show_params=True, detailed=False):
    """
    Visualize the cascaded tensor network structure showing actual tensor connectivity.
    
    Args:
        model: CascadedModel or cascade
        show_bonds: Show bond dimension details
        show_params: Show parameter counts
        detailed: Show individual tensor connections
    """
    # Get the cascade
    cascade = model.cascade if hasattr(model, 'cascade') else model
    
    print("=" * 100)
    print(f"CASCADED TENSOR NETWORK: {cascade.name}")
    print("=" * 100)
    
    # Overall architecture
    dims = [cascade.operators[0].config.input_dim]
    dims.extend([op.config.output_dim for op in cascade.operators])
    arch_str = " → ".join(str(d) for d in dims)
    print(f"\nArchitecture: {arch_str}")
    print(f"Compression ratio: {dims[0]/dims[-1]:.1f}x")
    
    # Layer-by-layer details
    print("\n" + "-" * 100)
    print("LAYER STRUCTURE:")
    print("-" * 100)
    
    total_params = 0
    
    for layer_idx, op in enumerate(cascade.operators):
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}: {op.config.input_dim} → {op.config.output_dim} ({op.operation_type})")
        print(f"{'='*60}")
        
        if hasattr(op, 'implementation'):
            impl = op.implementation
            
            # SMPO specific info
            if hasattr(impl, 'spacing'):
                spacing = impl.spacing
                print(f"Spacing: {spacing}")
                
                # Show which tensors have outputs
                output_positions = list(range(0, op.config.input_dim, spacing))
                print(f"Output positions: {output_positions}")
                
                if detailed:
                    # Show tensor connectivity pattern
                    print("\nTensor connectivity:")
                    for i in range(op.config.input_dim):
                        if i in output_positions:
                            print(f"  Tensor {i}: [Input k{i}] → [Output b{output_positions.index(i)}]")
                        else:
                            print(f"  Tensor {i}: [Input k{i}] → (no output)")
            
            # Tensor details
            if hasattr(impl, 'tensors') and show_params:
                tensors = impl.tensors
                layer_params = 0
                
                if detailed:
                    print(f"\nDetailed tensor information:")
                    for i, t in enumerate(tensors):
                        params = t.data.size
                        layer_params += params
                        
                        # Show indices for each tensor
                        print(f"  Tensor {i}:")
                        print(f"    Shape: {t.shape}")
                        print(f"    Indices: {t.inds}")
                        print(f"    Parameters: {params:,}")
                        
                        # Show bonds to adjacent tensors
                        if i > 0:
                            print(f"    Left bond: connects to tensor {i-1}")
                        if i < len(tensors) - 1:
                            print(f"    Right bond: connects to tensor {i+1}")
                else:
                    # Summarized view
                    shape_counts = {}
                    for t in tensors:
                        shape = tuple(t.shape)
                        shape_counts[shape] = shape_counts.get(shape, 0) + 1
                        layer_params += t.data.size
                    
                    print(f"\nTensor shapes in this layer:")
                    for shape, count in sorted(shape_counts.items()):
                        params_per_tensor = np.prod(shape)
                        total_for_shape = params_per_tensor * count
                        print(f"  - {count} tensors of shape {shape} = {total_for_shape:,} params")
                
                print(f"\nLayer {layer_idx} total parameters: {layer_params:,}")
                total_params += layer_params
    
    # Visual flow diagram showing actual SMPO spacing
    print("\n" + "-" * 100)
    print("TENSOR FLOW DIAGRAM:")
    print("-" * 100)
    
    for layer_idx, op in enumerate(cascade.operators):
        print(f"\nLayer {layer_idx}: {op.config.input_dim} → {op.config.output_dim}")
        
        if hasattr(op, 'implementation') and hasattr(op.implementation, 'spacing'):
            spacing = op.implementation.spacing
            
            # Calculate which tensors have outputs
            output_positions = list(range(0, op.config.input_dim, spacing))
            
            # Create input line showing ALL tensors
            input_line = []
            for i in range(op.config.input_dim):
                if i in output_positions:
                    input_line.append('●')  # Filled = has output
                else:
                    input_line.append('○')  # Hollow = no output
                if i < op.config.input_dim - 1:
                    input_line.append('─')
            
            # Create connection lines
            connection_line = []
            for i in range(op.config.input_dim):
                if i in output_positions:
                    connection_line.append('│')
                else:
                    connection_line.append(' ')
                if i < op.config.input_dim - 1:
                    connection_line.append(' ')
            
            # Create output line with proper spacing
            # Position outputs directly below their corresponding inputs
            output_line = [' '] * len(input_line)
            out_idx = 0
            for pos in output_positions:
                if pos * 2 < len(output_line):  # Account for the dashes
                    output_line[pos * 2] = '●'
                    out_idx += 1
            
            # Print the visual
            print(f"  Input:  {''.join(input_line)}")
            print(f"          {''.join(connection_line)}")
            print(f"  Output: {''.join(output_line)}")
            
            # Show the mapping clearly
            if detailed:
                print(f"\n  Tensor mapping:")
                for i, pos in enumerate(output_positions[:op.config.output_dim]):
                    print(f"    Input tensor {pos} → Output tensor {i}")
        else:
            # For non-SMPO layers (like expansion)
            print(f"  [{op.config.input_dim} tensors] → [{op.config.output_dim} tensors]")
    

    
    # Summary
    print("\n" + "-" * 100)
    print("SUMMARY:")
    print("-" * 100)
    print(f"Total layers: {len(cascade.operators)}")
    print(f"Total tensors: {sum(op.config.input_dim for op in cascade.operators)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Memory usage (float64): {total_params * 8 / 1024**2:.2f} MB")
    
    # Bottleneck analysis
    bottleneck_idx = cascade.bottleneck_index
    if bottleneck_idx > 0:
        bottleneck_dim = dims[bottleneck_idx]
        print(f"\nBottleneck: Layer {bottleneck_idx-1} → {bottleneck_dim} dimensions")
        print(f"Information compression: {dims[0]} → {bottleneck_dim} ({dims[0]/bottleneck_dim:.1f}x)")
    
    print("=" * 100)


def visualize_tensor_flow_detailed(cascade):
    """
    Create a detailed tensor flow diagram showing exact node connections.
    
    Shows:
    - ● Filled circles for tensors with outputs
    - ○ Hollow circles for tensors without outputs
    - Clear vertical alignment showing connections
    """
    print("\n" + "="*100)
    print("DETAILED TENSOR FLOW:")
    print("="*100)
    
    # Get all layer dimensions
    layer_dims = []
    for i, op in enumerate(cascade.operators):
        if i == 0:
            layer_dims.append(op.config.input_dim)
        layer_dims.append(op.config.output_dim)
    
    # Process each layer
    for layer_idx, op in enumerate(cascade.operators):
        input_dim = op.config.input_dim
        output_dim = op.config.output_dim
        
        print(f"\nLayer {layer_idx}: {input_dim} → {output_dim} ({op.operation_type})")
        
        if hasattr(op, 'implementation') and hasattr(op.implementation, 'spacing'):
            spacing = op.implementation.spacing
            
            # Calculate which positions have outputs
            output_positions = list(range(0, input_dim, spacing))[:output_dim]
            
            # Create the visualization
            # First, show input tensors
            print("\n  Input tensors:")
            print("  Position: ", end="")
            for i in range(input_dim):
                print(f"{i:3d}", end=" ")
            print()
            
            print("  Tensor:   ", end="")
            for i in range(input_dim):
                if i in output_positions:
                    print("  ● ", end="")  # Has output
                else:
                    print("  ○ ", end="")  # No output
            print()
            
            # Show connections
            print("            ", end="")
            for i in range(input_dim):
                if i in output_positions:
                    print("  │ ", end="")
                else:
                    print("    ", end="")
            print()
            
            # Show output tensors aligned with their inputs
            print("  Output tensors:")
            print("            ", end="")
            for i in range(input_dim):
                if i in output_positions:
                    idx = output_positions.index(i)
                    print(f"  {idx} ", end="")
                else:
                    print("    ", end="")
            print()
            
            print("            ", end="")
            for i in range(input_dim):
                if i in output_positions:
                    print("  ● ", end="")
                else:
                    print("    ", end="")
            print()
            
            # Summary
            print(f"\n  Spacing: {spacing} (every {spacing} tensors)")
            print(f"  Active tensors: {output_positions}")
            print(f"  Compression: {input_dim} → {output_dim} ({input_dim/output_dim:.1f}x)")
            
        else:
            # For non-SMPO layers
            print(f"  Standard layer: all {input_dim} inputs → all {output_dim} outputs")
    
    # Overall flow summary
    print("\n" + "="*100)
    print("OVERALL FLOW:")
    print("="*100)
    
    # Create a compact representation
    for layer_idx, op in enumerate(cascade.operators):
        if layer_idx == 0:
            print(f"Input: {op.config.input_dim} tensors")
        
        if hasattr(op, 'implementation') and hasattr(op.implementation, 'spacing'):
            spacing = op.implementation.spacing
            active = op.config.input_dim // spacing
            print(f"   ↓ Layer {layer_idx}: Keep every {spacing}-th tensor ({active} active)")
        else:
            print(f"   ↓ Layer {layer_idx}: {op.operation_type}")
        
        print(f"Layer {layer_idx} output: {op.config.output_dim} tensors")
    
    print("="*100)


def visualize_smpo_spacing(input_dim, output_dim, spacing):
    """
    Visualize SMPO spacing pattern in detail.
    
    Shows exactly which input tensors produce outputs.
    """
    print(f"\nSMPO SPACING VISUALIZATION")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Spacing: {spacing}")
    print("-" * 60)
    
    # Create visual grid
    print("\nInput tensors:  ", end="")
    for i in range(input_dim):
        if i % spacing == 0:
            print(f"{i:3d}", end="")
        else:
            print("  .", end="")
    print()
    
    print("Has output:     ", end="")
    for i in range(input_dim):
        if i % spacing == 0:
            print("  ↓", end="")
        else:
            print("   ", end="")
    print()
    
    print("Output indices: ", end="")
    out_idx = 0
    for i in range(input_dim):
        if i % spacing == 0:
            print(f"{out_idx:3d}", end="")
            out_idx += 1
        else:
            print("   ", end="")
    print()
    
    # Show contraction pattern
    print("\nContraction pattern:")
    print("- Tensors with outputs are contracted with their corresponding output indices")
    print("- Tensors without outputs are contracted with their neighbors")
    print(f"- Result: {input_dim} input tensors → {output_dim} output tensors")
    
    # ASCII art representation
    print("\nVisual representation:")
    print("┌" + "─" * (input_dim * 3 - 1) + "┐")
    
    # Input layer
    print("│", end="")
    for i in range(input_dim):
        if i % spacing == 0:
            print(" ●", end="")
        else:
            print(" ○", end="")
        if i < input_dim - 1:
            print("─", end="")
    print("│ Input MPS")
    
    # Connections
    print("│", end="")
    for i in range(input_dim):
        if i % spacing == 0:
            print(" │", end="")
        else:
            print("  ", end="")
        if i < input_dim - 1:
            print(" ", end="")
    print("│")
    
    # SMPO layer
    print("│", end="")
    for i in range(input_dim):
        if i % spacing == 0:
            print(" ■", end="")
        else:
            print(" □", end="")
        if i < input_dim - 1:
            print("─", end="")
    print("│ SMPO")
    
    # Output connections
    print("│", end="")
    out_idx = 0
    for i in range(input_dim):
        if i % spacing == 0:
            print(" │", end="")
            out_idx += 1
        else:
            print("  ", end="")
        if i < input_dim - 1:
            print(" ", end="")
    print("│")
    
    # Output layer
    print("└" + "─" * (output_dim * 3 - 1) + "┘")
    print(" ", end="")
    for i in range(output_dim):
        print(" ●", end="")
        if i < output_dim - 1:
            print("─", end="")
    print(" Output MPS")
    
    print("\nLegend:")
    print("● = Tensor with output index")
    print("○ = Tensor without output index")
    print("■ = SMPO tensor with lower index")
    print("□ = SMPO tensor without lower index")


def compare_cascade_models(model1, model2, names=("Model 1", "Model 2")):
    """Compare two cascade models side by side."""
    print("\nCOMPARING CASCADE ARCHITECTURES")
    print("=" * 80)
    
    for model, name in zip([model1, model2], names):
        print(f"\n{name}:")
        cascade = model.cascade if hasattr(model, 'cascade') else model
        
        dims = [cascade.operators[0].config.input_dim]
        dims.extend([op.config.output_dim for op in cascade.operators])
        
        arch = " → ".join(str(d) for d in dims)
        total_params = sum(t.data.size for t in model.tensors)
        
        print(f"  Architecture: {arch}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Compression: {dims[0]/dims[-1]:.1f}x")