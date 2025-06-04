"""
Autoencoder builder factory.

This module provides high-level functions to create complete autoencoder
architectures with proper dimension calculations and validation.
"""

from typing import List, Optional, Union, Tuple, Dict
import jax
import jax.numpy as jnp
from ..core.base import LayerConfig
from ..core.unified_operator import UnifiedCascadableOperator
from ..core.cascade import TensorNetworkCascade
from .dimension_calculator import DimensionCalculator


class AutoencoderBuilder:
    """
    Factory class for building tensor network autoencoders.
    
    Handles:
    - Dimension validation and suggestions
    - Automatic spacing calculations
    - Bond dimension optimization
    - Symmetric architecture generation
    """
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.dim_calc = DimensionCalculator(debug=False)
    
    def create_autoencoder(self,
                          layer_dims: List[int],
                          bond_dims: Optional[Union[int, List[int]]] = None,
                          cyclic: bool = False,
                          symmetric: bool = True,
                          initializer=None,
                          key=None,
                          validate_dims: bool = True,
                          **operator_kwargs) -> TensorNetworkCascade:
        """
        Create a complete autoencoder cascade.
        
        Args:
            layer_dims: Dimensions from input to bottleneck (e.g., [784, 256, 64, 16])
            bond_dims: Single value or list of bond dimensions
            cyclic: Whether to use cyclic boundary conditions
            symmetric: If True, create symmetric encoder-decoder architecture
            initializer: JAX initializer for tensors
            key: JAX random key
            validate_dims: Whether to validate/suggest dimensions for cyclic
            **operator_kwargs: Additional kwargs for operators
            
        Returns:
            TensorNetworkCascade representing the full autoencoder
        """
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (input and bottleneck)")
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)
        
        # Validate or suggest dimensions for cyclic
        if cyclic and validate_dims:
            layer_dims = self._validate_or_suggest_cyclic_dims(layer_dims)
        
        # Create full architecture
        if symmetric:
            full_dims = self._create_symmetric_architecture(layer_dims)
        else:
            full_dims = layer_dims
        
        if self.debug:
            print(f"\n[AUTOENCODER] Building architecture: {' → '.join(map(str, full_dims))}")
        
        # Handle bond dimensions
        bond_dims = self._prepare_bond_dimensions(full_dims, bond_dims, cyclic)
        
        # Create operators
        operators = self._create_operators(
            full_dims, bond_dims, cyclic, initializer, key, **operator_kwargs
        )
        
        # Create cascade
        name = f"{'Cyclic' if cyclic else 'Open'}Autoencoder_{full_dims[0]}to{min(full_dims)}"
        cascade = TensorNetworkCascade(operators, name=name, debug=self.debug)
        
        if self.debug:
            print(f"[AUTOENCODER] ✅ Created: {cascade}")
        
        return cascade
    
    def create_encoder_decoder_pair(self,
                                   encoder_dims: List[int],
                                   decoder_dims: Optional[List[int]] = None,
                                   **kwargs) -> Tuple[TensorNetworkCascade, TensorNetworkCascade]:
        """
        Create separate encoder and decoder cascades.
        
        Useful for:
        - Training encoder and decoder separately
        - Using different architectures for encoding/decoding
        - Transfer learning scenarios
        """
        if decoder_dims is None:
            decoder_dims = encoder_dims[-2::-1] + [encoder_dims[0]]
        
        # Create encoder
        encoder_key = kwargs.pop('key', jax.random.PRNGKey(42))
        encoder = self.create_autoencoder(
            encoder_dims,
            symmetric=False,
            key=encoder_key,
            **kwargs
        )
        
        # Create decoder
        decoder_key = jax.random.split(encoder_key)[0]
        decoder = self.create_autoencoder(
            decoder_dims,
            symmetric=False,
            key=decoder_key,
            **kwargs
        )
        
        return encoder, decoder
    
    def get_num_bond_dims_needed(self,
                                layer_dims: List[int],
                                symmetric: bool = True) -> int:
        """
        Calculate how many bond dimensions are needed for given architecture.
        
        Helpful for preparing custom bond dimension lists.
        """
        if symmetric:
            # Encoder + decoder (excluding repeated bottleneck)
            return 2 * (len(layer_dims) - 1)
        else:
            # Just the transitions between consecutive layers
            return len(layer_dims) - 1
    
    def suggest_architecture(self,
                           input_dim: int,
                           bottleneck_dim: int,
                           num_layers: int,
                           cyclic: bool = False,
                           compression_factor: Optional[float] = None) -> List[int]:
        """
        Suggest a good architecture given constraints.
        
        Args:
            input_dim: Input dimension
            bottleneck_dim: Desired bottleneck dimension
            num_layers: Number of layers from input to bottleneck
            cyclic: Whether to use cyclic boundaries
            compression_factor: Desired compression per layer (e.g., 0.5 for halving)
            
        Returns:
            List of dimensions [input, hidden1, ..., bottleneck]
        """
        if num_layers < 2:
            return [input_dim, bottleneck_dim]
        
        if cyclic:
            # Use dimension calculator for cyclic-compatible dims
            return self.dim_calc.suggest_cyclic_compatible_dims(
                input_dim, num_layers - 1, bottleneck_dim
            )
        
        # For non-cyclic, use geometric progression
        if compression_factor is None:
            compression_factor = (bottleneck_dim / input_dim) ** (1 / (num_layers - 1))
        
        dims = [input_dim]
        current = input_dim
        
        for i in range(1, num_layers - 1):
            current = int(current * compression_factor)
            dims.append(current)
        
        dims.append(bottleneck_dim)
        
        # Ensure monotonic decrease
        for i in range(1, len(dims)):
            if dims[i] >= dims[i-1]:
                dims[i] = max(1, dims[i-1] - 1)
        
        return dims
    
    def analyze_architecture(self, cascade: TensorNetworkCascade) -> Dict[str, any]:
        """
        Analyze an autoencoder architecture.
        
        Returns dict with:
        - dimensions: List of layer dimensions
        - compression_ratio: Overall compression
        - bottleneck_size: Size of bottleneck
        - is_symmetric: Whether encoder/decoder are symmetric
        - layer_types: Type of each layer (compression/expansion/identity)
        """
        dims = [cascade.operators[0].config.input_dim]
        dims.extend([op.config.output_dim for op in cascade.operators])
        
        layer_types = []
        for op in cascade.operators:
            if op.config.output_dim < op.config.input_dim:
                layer_types.append("compression")
            elif op.config.output_dim > op.config.input_dim:
                layer_types.append("expansion")
            else:
                layer_types.append("identity")
        
        bottleneck_idx = dims.index(min(dims))
        
        # Check symmetry
        is_symmetric = False
        if bottleneck_idx == len(dims) // 2:
            encoder_dims = dims[:bottleneck_idx+1]
            decoder_dims = dims[bottleneck_idx:]
            decoder_dims.reverse()
            is_symmetric = encoder_dims == decoder_dims
        
        return {
            'dimensions': dims,
            'compression_ratio': dims[0] / min(dims),
            'bottleneck_size': min(dims),
            'bottleneck_index': bottleneck_idx,
            'is_symmetric': is_symmetric,
            'layer_types': layer_types,
            'num_compression': layer_types.count('compression'),
            'num_expansion': layer_types.count('expansion'),
            'num_identity': layer_types.count('identity')
        }
    
    def _validate_or_suggest_cyclic_dims(self, layer_dims: List[int]) -> List[int]:
        """Validate dimensions for cyclic or suggest alternatives."""
        issues = self.dim_calc.check_cyclic_compatibility(layer_dims)
        
        if not issues:
            return layer_dims
        
        print(f"\n[AUTOENCODER] Original dimensions not cyclic-compatible")
        suggested = self.dim_calc.suggest_cyclic_compatible_dims(
            layer_dims[0], 
            len(layer_dims) - 1,
            layer_dims[-1]
        )
        
        print(f"[AUTOENCODER] Using suggested dimensions: {' → '.join(map(str, suggested))}")
        return suggested
    
    def _create_symmetric_architecture(self, encoder_dims: List[int]) -> List[int]:
        """Create symmetric encoder-decoder architecture."""
        # Encoder: input → bottleneck
        # Decoder: bottleneck → input (reversed, excluding bottleneck)
        decoder_dims = encoder_dims[-2::-1]
        return encoder_dims + decoder_dims
    
    def _prepare_bond_dimensions(self, 
                               full_dims: List[int],
                               bond_dims: Optional[Union[int, List[int]]],
                               cyclic: bool) -> List[int]:
        """Prepare bond dimensions for all layers."""
        num_layers = len(full_dims) - 1
        
        if bond_dims is None:
            # Auto-suggest bond dimensions
            return self.dim_calc.suggest_bond_dimensions(full_dims)
        elif isinstance(bond_dims, int):
            # Use same bond dimension for all layers
            return [bond_dims] * num_layers
        else:
            # User-provided list
            if len(bond_dims) != num_layers:
                # Helpful error message
                arch_str = " → ".join(str(d) for d in full_dims)
                raise ValueError(
                    f"Bond dimension mismatch:\n"
                    f"  Architecture: {arch_str}\n"
                    f"  Transitions: {num_layers}\n"
                    f"  Bond dims provided: {len(bond_dims)}\n"
                    f"  Bond dims expected: {num_layers}"
                )
            return bond_dims
    
    def _create_operators(self,
                         full_dims: List[int],
                         bond_dims: List[int],
                         cyclic: bool,
                         initializer,
                         key,
                         **operator_kwargs) -> List[UnifiedCascadableOperator]:
        """Create all operators for the cascade."""
        operators = []
        keys = jax.random.split(key, len(full_dims) - 1)
        
        for i in range(len(full_dims) - 1):
            config = LayerConfig(
                input_dim=full_dims[i],
                output_dim=full_dims[i + 1],
                bond_dim=bond_dims[i],
                cyclic=cyclic
            )
            
            op = UnifiedCascadableOperator(
                config=config,
                initializer=initializer,
                key=keys[i],
                debug=False,  # Less verbose
                **operator_kwargs
            )
            
            operators.append(op)
            
            if self.debug:
                op_type = "↓" if config.output_dim < config.input_dim else "↑"
                print(f"  Layer {i}: {full_dims[i]}→{full_dims[i+1]} ({op_type}), χ={bond_dims[i]}")
        
        return operators


def create_standard_autoencoder(input_dim: int,
                              compression_ratios: List[float] = [0.5, 0.25, 0.125],
                              cyclic: bool = False,
                              **kwargs) -> TensorNetworkCascade:
    """
    Convenience function to create standard autoencoder architectures.
    
    Examples:
        # MNIST-like: 784 → 392 → 196 → 98 → 196 → 392 → 784
        ae = create_standard_autoencoder(784, [0.5, 0.25, 0.125])
        
        # Custom ratios
        ae = create_standard_autoencoder(1024, [0.5, 0.125])  # 1024 → 512 → 128 → 512 → 1024
    """
    builder = AutoencoderBuilder()
    
    # Calculate dimensions from compression ratios
    dims = [input_dim]
    current = input_dim
    for ratio in compression_ratios:
        current = int(current * ratio)
        dims.append(current)
    
    return builder.create_autoencoder(dims, cyclic=cyclic, **kwargs)