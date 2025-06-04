"""
Unified operator that handles both compression and expansion.

This module provides a single operator class that automatically chooses
the right tensor network structure based on input/output dimensions.
"""

from typing import Optional, Union, Any
import jax
import jax.numpy as jnp
from tn4ml.models.smpo import SMPO_initialize
from tn4ml.models.mpo import MPO_initialize
import quimb.tensor as qtn

from .base import CascadableOperator, LayerConfig, debug_timer, debug_trace


class UnifiedCascadableOperator(CascadableOperator):
    """
    Unified operator that automatically handles compression and expansion.
    
    Based on input/output dimensions, it chooses:
    - Compression (out < in): Uses SMPO with spacing
    - Expansion (out > in): Uses MPO-like structure with shared inputs
    - Identity (out = in): Uses regular MPO
    """
    
    def __init__(self,
                 config: LayerConfig,
                 initializer=None,
                 key=None,
                 debug: bool = False,
                 debug_level: int = 0,
                 **kwargs):
        """Initialize unified operator."""
        super().__init__(debug=debug, debug_level=debug_level)
        
        self.config = config
        self.operation_type = self._determine_operation_type()
        
        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        if self.debug:
            print(f"[UNIFIED] Creating {self.operation_type} operator: "
                  f"{config.input_dim}→{config.output_dim}")
        
        # Create appropriate implementation
        if self.operation_type == "compression":
            self.implementation = self._create_compression_layer(
                config, initializer, key, **kwargs
            )
        elif self.operation_type == "expansion":
            self.implementation = self._create_expansion_layer(
                config, initializer, key, **kwargs
            )
        else:  # identity
            self.implementation = self._create_identity_layer(
                config, initializer, key, **kwargs
            )
        
        # Cache for debugging
        self._last_input_shape = None
        self._last_output_shape = None
        self._application_count = 0
    
    def _determine_operation_type(self) -> str:
        """Determine if this is compression, expansion, or identity."""
        if self.config.output_dim < self.config.input_dim:
            return "compression"
        elif self.config.output_dim > self.config.input_dim:
            return "expansion"
        else:
            return "identity"
    
    def _create_compression_layer(self, config, initializer, key, **kwargs):
        """Create compression layer using SMPO."""
        # Calculate spacing if not provided
        if config.spacing is None:
            from ..builders.dimension_calculator import DimensionCalculator
            calc = DimensionCalculator(debug=False)
            spacing_result = calc.calculate_optimal_spacing(
                config.input_dim, config.output_dim, config.cyclic
            )
            spacing = spacing_result.spacing
            
            if self.debug:
                print(f"[COMPRESSION] Auto-calculated spacing: {spacing}")
        else:
            spacing = config.spacing
        
        # Handle non-uniform spacing
        if isinstance(spacing, list):
            # For non-uniform spacing, we need to handle it differently
            # SMPO_initialize might not support 'spacings' directly
            # For now, just use the first spacing value
            if self.debug:
                print(f"[WARNING] Non-uniform spacing not fully supported yet, using first value")
            spacing = spacing[0]
        
        # Create SMPO
        smpo = SMPO_initialize(
            L=config.input_dim,  # Number of tensors = number of inputs
            initializer=initializer,
            key=key,
            bond_dim=config.bond_dim,
            phys_dim=config.phys_dim,
            cyclic=config.cyclic,
            add_identity=config.add_identity,
            boundary='pbc' if config.cyclic else 'obc',
            spacing=spacing,  # Always use 'spacing', not 'spacings'
            **kwargs
        )
        
        # Verify output count
        actual_outputs = len(list(smpo.lower_inds))
        if actual_outputs != config.output_dim:
            print(f"[WARNING] SMPO created with {actual_outputs} outputs, "
                  f"expected {config.output_dim}")
        
        return smpo
    
    def _create_expansion_layer(self, config, initializer, key, **kwargs):
        """
        Create expansion layer using MPO-like structure.
        
        For expansion, we need N tensors (one per output) where
        each input connects to multiple tensors.
        """
        if self.debug:
            print(f"[EXPANSION] Creating {config.output_dim} tensors "
                  f"for {config.input_dim} inputs")
        
        # For now, create a placeholder structure
        # In a real implementation, this would create tensors where:
        # - Number of tensors = output_dim
        # - Each tensor has access to all input_dim inputs
        # - The connectivity pattern determines the expansion
        
        # Calculate which outputs each input connects to
        connections = self._calculate_expansion_connections(
            config.input_dim, config.output_dim
        )
        
        # Build tensor network
        tensors = []
        for i in range(config.output_dim):
            # Each output tensor
            # Shape: (left_bond, right_bond, phys_up, phys_down)
            shape = self._get_tensor_shape(i, config)
            
            # Initialize tensor
            tensor_key = jax.random.split(key, config.output_dim)[i]
            data = initializer(tensor_key, shape)
            
            # Create indices
            inds = self._get_tensor_indices(i, config, connections)
            
            # Create quimb tensor
            tensor = qtn.Tensor(data=data, inds=inds, tags=[f'I{i}'])
            tensors.append(tensor)
        
        # Create TensorNetwork
        tn = qtn.TensorNetwork(tensors)
        
        # Add expansion-specific attributes
        tn.operation_type = 'expansion'
        tn.input_dim = config.input_dim
        tn.output_dim = config.output_dim
        tn.connections = connections
        
        return tn
    
    def _create_identity_layer(self, config, initializer, key, **kwargs):
        """Create identity layer using tensors."""
        if self.debug:
            print(f"[IDENTITY] Creating identity transformation")
        
        # Create actual tensors for identity operation
        return self._create_identity_placeholder(config)
    
    def _calculate_expansion_connections(self, input_dim: int, 
                                       output_dim: int) -> dict:
        """
        Calculate which inputs connect to which outputs for expansion.
        
        Returns:
            Dict mapping input indices to list of output indices
        """
        connections = {}
        
        # Simple strategy: distribute inputs evenly across outputs
        outputs_per_input = output_dim / input_dim
        
        for i in range(input_dim):
            start = int(i * outputs_per_input)
            end = int((i + 1) * outputs_per_input)
            connections[i] = list(range(start, end))
        
        # Ensure all outputs are covered
        all_outputs = set()
        for outputs in connections.values():
            all_outputs.update(outputs)
        
        if len(all_outputs) < output_dim:
            # Distribute remaining outputs
            remaining = set(range(output_dim)) - all_outputs
            for i, out_idx in enumerate(remaining):
                input_idx = i % input_dim
                connections[input_idx].append(out_idx)
        
        return connections
    
    def _get_tensor_shape(self, idx: int, config: LayerConfig) -> tuple:
        """Get shape for tensor at given index."""
        # Simplified - real implementation would be more sophisticated
        if self.operation_type == "compression":
            return (config.bond_dim, config.bond_dim, 
                    config.phys_dim[0], config.phys_dim[1])
        else:  # expansion
            return (config.bond_dim, config.bond_dim,
                    config.phys_dim[0], config.phys_dim[1])
    
    def _get_tensor_indices(self, idx: int, config: LayerConfig, 
                          connections: dict = None) -> list:
        """Get indices for tensor at given position."""
        # Simplified - real implementation would handle proper connectivity
        if self.operation_type == "compression":
            return [f'vL{idx}', f'vR{idx}', f'k{idx}', f'b{idx}']
        else:  # expansion
            # Input indices based on connections
            input_idx = self._find_input_for_output(idx, connections)
            return [f'vL{idx}', f'vR{idx}', f'k{input_idx}', f'b{idx}']
    
    def _find_input_for_output(self, output_idx: int, connections: dict) -> int:
        """Find which input connects to given output."""
        for input_idx, outputs in connections.items():
            if output_idx in outputs:
                return input_idx
        return 0  # Fallback
    
    def _create_identity_placeholder(self, config):
        """Create placeholder for identity operation."""
        # For identity, we need actual tensors for training
        # Create a simple pass-through tensor network
        import jax.numpy as jnp
        
        tensors = []
        for i in range(config.input_dim):
            # Create identity-like tensor
            # Shape: (bond_left, bond_right, phys_up, phys_down)
            shape = (config.bond_dim, config.bond_dim, 
                    config.phys_dim[0], config.phys_dim[1])
            
            # Initialize as near-identity
            data = jnp.zeros(shape)
            # Set diagonal elements
            min_dim = min(shape)
            for j in range(min_dim):
                data = data.at[j, j, j % config.phys_dim[0], j % config.phys_dim[1]].set(1.0)
            
            # Create tensor with proper indices
            inds = [f'vL{i}', f'vR{i}', f'k{i}', f'b{i}']
            tensor = qtn.Tensor(data=data, inds=inds, tags=[f'I{i}'])
            tensors.append(tensor)
        
        # Create TensorNetwork
        tn = qtn.TensorNetwork(tensors)
        tn.operation_type = 'identity'
        tn.input_dim = config.input_dim
        tn.output_dim = config.output_dim
        
        return tn
    
    @debug_timer
    @debug_trace
    def apply(self, input_mps):
        """Apply the operator to input MPS."""
        self._application_count += 1
        
        if self.debug:
            print(f"\n[APPLY #{self._application_count}] {self} ({self.operation_type})")
        
        if self.operation_type == "compression":
            # Use SMPO apply
            return self.implementation.apply(input_mps)
        elif self.operation_type == "expansion":
            # Custom expansion logic
            return self._apply_expansion(input_mps)
        else:
            # Identity
            return self._apply_identity(input_mps)
    
    def _apply_expansion(self, input_mps):
        """Apply expansion operation."""
        if self.debug:
            print(f"[EXPANSION] Applying expansion: {self.config.input_dim}→{self.config.output_dim}")
        
        # Placeholder - real implementation would:
        # 1. Contract input MPS with expansion tensors
        # 2. Handle the index routing properly
        # 3. Return expanded MPS
        
        # For now, just return input with warning
        print(f"[WARNING] Expansion apply not yet fully implemented")
        return input_mps
    
    def _apply_identity(self, input_mps):
        """Apply identity operation."""
        if self.debug:
            print(f"[IDENTITY] Applying identity transformation")
        
        # For now, just pass through
        # Real implementation would apply the identity tensors
        return input_mps
    
    def get_config(self) -> LayerConfig:
        """Return configuration."""
        return self.config
    
    def __repr__(self):
        """String representation."""
        cyclic_str = "↻" if self.config.cyclic else "→"
        type_str = {"compression": "↓", "expansion": "↑", "identity": "="}[self.operation_type]
        return (f"Unified{type_str}({self.config.input_dim}{cyclic_str}"
                f"{self.config.output_dim}, χ={self.config.bond_dim})")