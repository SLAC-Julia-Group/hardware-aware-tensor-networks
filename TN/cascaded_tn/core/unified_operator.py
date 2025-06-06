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
from .expansion import ExpansionOperator


class UnifiedCascadableOperator(CascadableOperator):
    """
    Unified operator that automatically handles compression and expansion.
    
    Based on input/output dimensions, it chooses:
    - Compression (out < in): Uses SMPO with spacing
    - Expansion (out > in): Uses ExpansionOperator
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
            # Use the new ExpansionOperator
            self.implementation = ExpansionOperator(
                config=config,
                initializer=initializer,
                key=key,
                debug=debug,
                debug_level=debug_level,
                **kwargs
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
            if self.debug:
                print(f"[WARNING] Non-uniform spacing not fully supported yet, using first value")
            spacing = spacing[0]
        
        # Extract boundary from kwargs to avoid duplicate
        boundary = kwargs.pop('boundary', 'pbc' if config.cyclic else 'obc')
        
        # Create SMPO
        smpo = SMPO_initialize(
            L=config.input_dim,
            initializer=initializer,
            key=key,
            bond_dim=config.bond_dim,
            phys_dim=config.phys_dim,
            cyclic=config.cyclic,
            add_identity=config.add_identity,
            boundary=boundary,
            spacing=spacing,
            **kwargs  # Pass remaining kwargs
        )
        
        # Verify output count
        actual_outputs = len(list(smpo.lower_inds))
        if actual_outputs != config.output_dim:
            print(f"[WARNING] SMPO created with {actual_outputs} outputs, "
                f"expected {config.output_dim}")
        
        return smpo

    def _create_identity_layer(self, config, initializer, key, **kwargs):
        """Create identity layer using MPO."""
        if self.debug:
            print(f"[IDENTITY] Creating identity transformation")
        
        # For identity, create an MPO with identity-like structure
        # This is a simplified version - might need more sophisticated implementation
        
        # Create MPO-like structure
        tensors = []
        keys = jax.random.split(key, config.input_dim)
        
        for i in range(config.input_dim):
            # Shape: (left_bond, right_bond, phys_up, phys_down)
            if i == 0:
                shape = (1, config.bond_dim, config.phys_dim[0], config.phys_dim[1])
            elif i == config.input_dim - 1:
                shape = (config.bond_dim, 1, config.phys_dim[0], config.phys_dim[1])
            else:
                shape = (config.bond_dim, config.bond_dim, config.phys_dim[0], config.phys_dim[1])
            
            # Initialize with identity-like structure
            data = initializer(keys[i], shape)
            
            # Add identity bias
            if config.add_identity:
                # Make diagonal elements stronger
                if shape[0] == shape[1]:  # Square virtual bonds
                    for j in range(min(shape[0], shape[2], shape[3])):
                        data = data.at[j, j, j % shape[2], j % shape[3]].add(1.0)
            
            # Create indices
            inds = [f'vL{i}', f'vR{i}', f'k{i}', f'b{i}']
            
            # Create tensor
            tensor = qtn.Tensor(data=data, inds=inds, tags=[f'I{i}'])
            tensors.append(tensor)
        
        # Create TensorNetwork
        tn = qtn.TensorNetwork(tensors)
        tn.operation_type = 'identity'
        tn.input_dim = config.input_dim
        tn.output_dim = config.output_dim
        
        # Add required attributes for compatibility
        tn.lower_inds = [f'b{i}' for i in range(config.output_dim)]
        tn.upper_inds = [f'k{i}' for i in range(config.input_dim)]
        
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
            # Use ExpansionOperator apply
            return self.implementation.apply(input_mps)
        else:
            # Identity - contract with MPO-like structure
            return self._apply_identity(input_mps)
    
    def _apply_identity(self, input_mps):
        """Apply identity operation."""
        if self.debug:
            print(f"[IDENTITY] Applying identity transformation")
        
        # Contract input MPS with identity MPO
        input_copy = input_mps.copy()
        identity_copy = self.implementation.copy()
        
        # Align indices
        for i in range(self.config.input_dim):
            # Find the physical index in input tensor
            input_tensor = input_copy.tensors[i]
            
            # The physical index is usually the one with smallest dimension
            # or the one that's not a bond index
            phys_idx = None
            for idx in input_tensor.inds:
                if 'bond' not in idx and 'v' not in idx:
                    phys_idx = idx
                    break
            
            if phys_idx is None:
                # Fallback: smallest dimension
                phys_idx = min(input_tensor.inds, key=lambda x: input_tensor.ind_size(x))
            
            # Reindex to match our k indices
            input_tensor.reindex_({phys_idx: f'k{i}'})
        
        # Contract
        result = input_copy | identity_copy
        
        # Contract all k indices
        for i in range(self.config.input_dim):
            k_ind = f'k{i}'
            if k_ind in result.ind_map:
                result.contract_ind(k_ind)
        
        # Convert back to MPS
        output_arrays = []
        for i in range(self.config.output_dim):
            tensor = result[f'I{i}']
            data = tensor.data
            
            # Ensure correct MPS shape
            if len(data.shape) > 3:
                data = jnp.squeeze(data)
            
            output_arrays.append(data)
        
        from quimb.tensor.tensor_1d import MatrixProductState
        output_mps = MatrixProductState(output_arrays, shape='lrp')
        
        if self.debug:
            print(f"[IDENTITY] Output norm: {output_mps.norm():.6f}")
        
        return output_mps
    
    def get_config(self) -> LayerConfig:
        """Return configuration."""
        return self.config
    
    @property
    def tensors(self):
        """Access to underlying tensors for training."""
        if hasattr(self.implementation, 'tensors'):
            return self.implementation.tensors
        elif hasattr(self.implementation, 'tn') and hasattr(self.implementation.tn, 'tensors'):
            return self.implementation.tn.tensors
        else:
            # Fallback for identity/custom implementations
            return list(self.implementation.tensors) if hasattr(self.implementation, '__iter__') else []
    
    def __repr__(self):
        """String representation."""
        cyclic_str = "↻" if self.config.cyclic else "→"
        type_str = {"compression": "↓", "expansion": "↑", "identity": "="}[self.operation_type]
        return (f"Unified{type_str}({self.config.input_dim}{cyclic_str}"
                f"{self.config.output_dim}, χ={self.config.bond_dim})")