"""
Encoder builder factory.

This module provides high-level functions to create cascaded tensor network
encoder architectures with proper dimension calculations and validation.
"""

from typing import List, Optional, Union, Tuple, Dict
import jax
import jax.numpy as jnp
from ..core.base import LayerConfig
from ..core.unified_operator import UnifiedCascadableOperator
from ..core.cascade import TensorNetworkCascade
from .dimension_calculator import DimensionCalculator


class EncoderBuilder:
    """
    Factory class for building cascaded tensor network encoders.

    Handles:
    - Dimension validation and suggestions
    - Automatic spacing calculations
    - Bond dimension optimization
    - Architecture generation
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.dim_calc = DimensionCalculator(debug=False)

    def create_encoder(self,
                       layer_dims: List[int],
                       bond_dims: Optional[Union[int, List[int]]] = None,
                       cyclic: bool = False,
                       initializer=None,
                       key=None,
                       validate_dims: bool = True,
                       enable_relu: Optional[Union[bool, List[int]]] = None,
                       output_positions: Optional[Union[str, List[List[int]]]] = None,
                       **operator_kwargs) -> TensorNetworkCascade:
        """
        Create a cascaded encoder.

        Args:
            layer_dims: Dimensions from input to output (e.g., [784, 256, 64, 16])
            bond_dims: Single value or list of bond dimensions
            cyclic: Whether to use cyclic boundary conditions
            initializer: JAX initializer for tensors
            key: JAX random key
            validate_dims: Whether to validate/suggest dimensions for cyclic
            enable_relu: Enable ReLU between layers (bool or list of layer indices)
            output_positions: Control output placement for each layer:
                - None: Use automatic uniform spacing (default)
                - 'center': Place outputs symmetrically centered in each layer
                - List[List[int]]: Explicit output positions per layer
            **operator_kwargs: Additional kwargs for operators

        Returns:
            TensorNetworkCascade representing the encoder
        """
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (input and output)")

        if key is None:
            key = jax.random.PRNGKey(42)

        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)

        # Validate or suggest dimensions for cyclic
        if cyclic and validate_dims:
            layer_dims = self._validate_or_suggest_cyclic_dims(layer_dims)

        if self.debug:
            print(f"\n[ENCODER] Building architecture: {' → '.join(map(str, layer_dims))}")

        # Handle bond dimensions
        bond_dims = self._prepare_bond_dimensions(layer_dims, bond_dims, cyclic)

        # Process ReLU configuration
        relu_configs = [False] * (len(layer_dims) - 1)
        if enable_relu is not None:
            if isinstance(enable_relu, bool) and enable_relu:
                relu_configs = [True] * (len(layer_dims) - 1)
            elif isinstance(enable_relu, list):
                for layer_idx in enable_relu:
                    if 0 <= layer_idx < len(relu_configs):
                        relu_configs[layer_idx] = True

        # Process output positions configuration
        output_inds_per_layer = self._prepare_output_positions(layer_dims, output_positions)

        # Handle physical dimensions
        if 'phys_dims' in operator_kwargs:
            phys_dims_list = operator_kwargs['phys_dims']
            if not isinstance(phys_dims_list, list):
                raise ValueError("phys_dims must be a list")
            num_layers = len(layer_dims) - 1
            if len(phys_dims_list) != len(layer_dims):
                raise ValueError(
                    f"phys_dims must have {len(layer_dims)} elements for {len(layer_dims)} layer dimensions, "
                    f"got {len(phys_dims_list)}"
                )

        # Create operators
        operators = self._create_operators(
            layer_dims, bond_dims, relu_configs, output_inds_per_layer,
            cyclic, initializer, key, **operator_kwargs
        )

        # Create cascade
        name = f"{'Cyclic' if cyclic else 'Open'}Encoder_{layer_dims[0]}to{layer_dims[-1]}"
        cascade = TensorNetworkCascade(operators, name=name, debug=self.debug)

        if self.debug:
            print(f"[ENCODER] Created: {cascade}")

        return cascade

    def get_num_bond_dims_needed(self, layer_dims: List[int]) -> int:
        """
        Calculate how many bond dimensions are needed for given architecture.

        Helpful for preparing custom bond dimension lists.
        """
        return len(layer_dims) - 1

    def suggest_architecture(self,
                             input_dim: int,
                             output_dim: int,
                             num_layers: int,
                             cyclic: bool = False,
                             compression_factor: Optional[float] = None) -> List[int]:
        """
        Suggest a good architecture given constraints.

        Args:
            input_dim: Input dimension
            output_dim: Desired output dimension
            num_layers: Number of layers from input to output
            cyclic: Whether to use cyclic boundaries
            compression_factor: Desired compression per layer (e.g., 0.5 for halving)

        Returns:
            List of dimensions [input, hidden1, ..., output]
        """
        if num_layers < 2:
            return [input_dim, output_dim]

        if cyclic:
            return self.dim_calc.suggest_cyclic_compatible_dims(
                input_dim, num_layers - 1, output_dim
            )

        if compression_factor is None:
            compression_factor = (output_dim / input_dim) ** (1 / (num_layers - 1))

        dims = [input_dim]
        current = input_dim

        for i in range(1, num_layers - 1):
            current = int(current * compression_factor)
            dims.append(current)

        dims.append(output_dim)

        # Ensure monotonic decrease
        for i in range(1, len(dims)):
            if dims[i] >= dims[i-1]:
                dims[i] = max(1, dims[i-1] - 1)

        return dims

    def analyze_architecture(self, cascade: TensorNetworkCascade) -> Dict[str, any]:
        """
        Analyze an encoder architecture.

        Returns dict with:
        - dimensions: List of layer dimensions
        - compression_ratio: Overall compression
        - output_size: Size of final output
        - layer_types: Type of each layer (compression/identity)
        """
        dims = [cascade.operators[0].config.input_dim]
        dims.extend([op.config.output_dim for op in cascade.operators])

        layer_types = []
        for op in cascade.operators:
            if op.config.output_dim < op.config.input_dim:
                layer_types.append("compression")
            else:
                layer_types.append("identity")

        return {
            'dimensions': dims,
            'compression_ratio': dims[0] / dims[-1],
            'output_size': dims[-1],
            'layer_types': layer_types,
            'num_compression': layer_types.count('compression'),
            'num_identity': layer_types.count('identity')
        }

    def _validate_or_suggest_cyclic_dims(self, layer_dims: List[int]) -> List[int]:
        """Validate dimensions for cyclic or suggest alternatives."""
        issues = self.dim_calc.check_cyclic_compatibility(layer_dims)

        if not issues:
            return layer_dims

        print(f"\n[ENCODER] Original dimensions not cyclic-compatible")
        suggested = self.dim_calc.suggest_cyclic_compatible_dims(
            layer_dims[0],
            len(layer_dims) - 1,
            layer_dims[-1]
        )

        print(f"[ENCODER] Using suggested dimensions: {' → '.join(map(str, suggested))}")
        return suggested

    def _prepare_output_positions(self,
                                  layer_dims: List[int],
                                  output_positions: Optional[Union[str, List[List[int]]]]) -> List[Optional[List[int]]]:
        """
        Prepare output_inds for each layer based on output_positions configuration.

        Args:
            layer_dims: Full architecture dimensions
            output_positions: User specification ('center', explicit list, or None)

        Returns:
            List of output_inds for each layer (None means use spacing)
        """
        num_layers = len(layer_dims) - 1

        if output_positions is None:
            return [None] * num_layers

        if isinstance(output_positions, str):
            if output_positions == 'center':
                output_inds_list = []
                for i in range(num_layers):
                    input_dim = layer_dims[i]
                    output_dim = layer_dims[i + 1]
                    centered = self.dim_calc.calculate_centered_positions(input_dim, output_dim)
                    output_inds_list.append(centered)

                    if self.debug:
                        print(f"  Layer {i}: centered output_inds = {centered}")

                return output_inds_list
            else:
                raise ValueError(f"Unknown output_positions strategy: '{output_positions}'. "
                            f"Use 'center' or provide explicit list.")

        elif isinstance(output_positions, list):
            if len(output_positions) != num_layers:
                raise ValueError(
                    f"output_positions must have {num_layers} elements for {num_layers} layers, "
                    f"got {len(output_positions)}"
                )
            return output_positions

        else:
            raise ValueError(f"output_positions must be None, 'center', or a list, "
                        f"got {type(output_positions)}")

    def _prepare_bond_dimensions(self,
                                 layer_dims: List[int],
                                 bond_dims: Optional[Union[int, List[int]]],
                                 cyclic: bool) -> List[int]:
        """Prepare bond dimensions for all layers."""
        num_layers = len(layer_dims) - 1

        if bond_dims is None:
            return self.dim_calc.suggest_bond_dimensions(layer_dims)
        elif isinstance(bond_dims, int):
            return [bond_dims] * num_layers
        else:
            if len(bond_dims) != num_layers:
                arch_str = " → ".join(str(d) for d in layer_dims)
                raise ValueError(
                    f"Bond dimension mismatch:\n"
                    f"  Architecture: {arch_str}\n"
                    f"  Transitions: {num_layers}\n"
                    f"  Bond dims provided: {len(bond_dims)}\n"
                    f"  Bond dims expected: {num_layers}"
                )
            return bond_dims

    def _create_operators(self,
                          layer_dims: List[int],
                          bond_dims: List[int],
                          relu_configs: List[bool],
                          output_inds_per_layer: List[Optional[List[int]]],
                          cyclic: bool,
                          initializer,
                          key,
                          **operator_kwargs) -> List[UnifiedCascadableOperator]:
        """Create all operators for the cascade."""
        operators = []
        keys = jax.random.split(key, len(layer_dims) - 1)

        # Extract config-specific kwargs
        phys_dims = operator_kwargs.pop('phys_dims', None)
        phys_dim = operator_kwargs.pop('phys_dim', (2, 2))
        add_identity = operator_kwargs.pop('add_identity', False)

        num_layers = len(layer_dims) - 1

        if phys_dims is not None:
            if not isinstance(phys_dims, list):
                raise ValueError("phys_dims must be a list of dimensions")
            if len(phys_dims) != num_layers + 1:
                raise ValueError(
                    f"phys_dims must have {num_layers + 1} dimensions for {num_layers} layers, "
                    f"got {len(phys_dims)}"
                )
            layer_phys_dims = [(phys_dims[i], phys_dims[i+1]) for i in range(num_layers)]
        else:
            layer_phys_dims = [phys_dim] * num_layers

        for i in range(num_layers):
            config = LayerConfig(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                bond_dim=bond_dims[i],
                cyclic=cyclic,
                phys_dim=layer_phys_dims[i],
                add_identity=add_identity,
                enable_relu=relu_configs[i],
                output_inds=output_inds_per_layer[i]
            )

            op = UnifiedCascadableOperator(
                config=config,
                initializer=initializer,
                key=keys[i],
                debug=False,
                **operator_kwargs
            )

            operators.append(op)

            if self.debug:
                op_type = "↓" if config.output_dim < config.input_dim else "="
                pos_str = f", pos={config.output_inds}" if config.output_inds else ""
                print(f"  Layer {i}: {layer_dims[i]}→{layer_dims[i+1]} ({op_type}), "
                    f"χ={bond_dims[i]}, φ={layer_phys_dims[i]}{pos_str}")

        return operators


def create_standard_encoder(input_dim: int,
                             compression_ratios: List[float] = [0.5, 0.25, 0.125],
                             cyclic: bool = False,
                             phys_dims: Optional[List[int]] = None,
                             **kwargs) -> TensorNetworkCascade:
    """
    Convenience function to create standard encoder architectures.

    Args:
        input_dim: Input dimension
        compression_ratios: List of compression ratios for each layer
        cyclic: Whether to use cyclic boundary conditions
        phys_dims: Optional list of physical dimensions [m0, m1, m2, ...] for sequential matching
        **kwargs: Additional arguments passed to create_encoder

    Examples:
        # 784 → 392 → 196 → 98
        enc = create_standard_encoder(784, [0.5, 0.25, 0.125])

        # Custom ratios
        enc = create_standard_encoder(1024, [0.5, 0.125])

        # With custom physical dimensions
        enc = create_standard_encoder(784, [0.5, 0.25], phys_dims=[2, 3, 4])
    """
    builder = EncoderBuilder()

    dims = [input_dim]
    current = input_dim
    for ratio in compression_ratios:
        current = int(current * ratio)
        dims.append(current)

    if phys_dims is not None:
        kwargs['phys_dims'] = phys_dims

    return builder.create_encoder(dims, cyclic=cyclic, **kwargs)
