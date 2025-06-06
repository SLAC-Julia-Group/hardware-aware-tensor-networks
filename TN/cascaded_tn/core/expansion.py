"""
expansion.py - Expansion operator for cascaded tensor networks.

This module implements expansion operations (few inputs → many outputs)
which are not natively supported by SMPO.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState

from .base import CascadableOperator, LayerConfig, debug_timer, debug_trace


class ExpansionOperator(CascadableOperator):
    """
    Expansion operator that reverses the SMPO concept.
    
    Instead of having many tensors with few outputs, we have:
    - Few input indices that get distributed to many output tensors
    - Each input connects to multiple output positions
    
    Architecture:
    - Input MPS has N sites
    - Output MPS has M sites (M > N)
    - We create M tensors where each knows which input(s) to read from
    """
    
    def __init__(self,
                 config: LayerConfig,
                 initializer=None,
                 key=None,
                 debug: bool = False,
                 debug_level: int = 0,
                 **kwargs):
        """Initialize expansion operator."""
        super().__init__(debug=debug, debug_level=debug_level)
        
        if config.output_dim <= config.input_dim:
            raise ValueError(f"ExpansionOperator requires output > input, "
                           f"got {config.input_dim}→{config.output_dim}")
        
        self.config = config
        
        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Calculate connectivity pattern
        self.connectivity = self._calculate_connectivity()
        
        if self.debug:
            print(f"[EXPANSION] Creating {config.input_dim}→{config.output_dim}")
            print(f"[EXPANSION] Connectivity pattern: {self.connectivity}")
        
        # Create the expansion tensor network
        self._create_expansion_network(initializer, key, **kwargs)
    
    def _calculate_connectivity(self) -> Dict[int, List[int]]:
        """
        Calculate which outputs each input connects to.
        
        Returns mapping: input_idx -> [output_indices]
        """
        input_dim = self.config.input_dim
        output_dim = self.config.output_dim
        
        # Strategy: Distribute outputs evenly among inputs
        outputs_per_input = output_dim / input_dim
        
        connectivity = {}
        for i in range(input_dim):
            start = int(i * outputs_per_input)
            end = int((i + 1) * outputs_per_input)
            connectivity[i] = list(range(start, end))
        
        # Handle remainder outputs
        assigned = sum(len(v) for v in connectivity.values())
        if assigned < output_dim:
            # Distribute remaining outputs round-robin
            for i in range(assigned, output_dim):
                input_idx = (i - assigned) % input_dim
                connectivity[input_idx].append(i)
        
        return connectivity
    
    def _create_expansion_network(self, initializer, key, **kwargs):
        """Create the expansion tensor network."""
        output_dim = self.config.output_dim
        bond_dim = self.config.bond_dim
        phys_dim = self.config.phys_dim
        
        # Create tensors for each output position
        tensors = []
        keys = jax.random.split(key, output_dim)
        
        for out_idx in range(output_dim):
            # Find which input this output reads from
            input_idx = self._find_input_for_output(out_idx)
            
            # Determine tensor shape based on position
            if out_idx == 0:
                # First tensor
                shape = (1, bond_dim, phys_dim[0], phys_dim[1])
            elif out_idx == output_dim - 1:
                # Last tensor
                shape = (bond_dim, 1, phys_dim[0], phys_dim[1])
            else:
                # Middle tensors
                shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
            
            # Initialize tensor data
            data = initializer(keys[out_idx], shape)
            
            # Create indices
            # Virtual bonds connect adjacent output tensors
            # Physical indices: upper connects to input, lower is output
            inds = []
            
            # Left virtual bond
            if out_idx == 0:
                if self.config.cyclic:
                    inds.append(f'vR{output_dim-1}')
            else:
                inds.append(f'vR{out_idx-1}')
            
            # Right virtual bond
            if out_idx == output_dim - 1:
                if self.config.cyclic:
                    inds.append(f'vR{out_idx}')
            else:
                inds.append(f'vR{out_idx}')
            
            # Physical indices
            inds.append(f'k{input_idx}')  # Upper: connects to input
            inds.append(f'b{out_idx}')    # Lower: output index
            
            # Create tensor
            tensor = qtn.Tensor(data=data, inds=inds, tags=[f'O{out_idx}'])
            tensors.append(tensor)
        
        # Store as TensorNetwork
        self.tn = qtn.TensorNetwork(tensors)
        
        # Store metadata
        self.tn.input_dim = self.config.input_dim
        self.tn.output_dim = self.config.output_dim
        self.tn.connectivity = self.connectivity
        
        if self.debug:
            print(f"[EXPANSION] Created {len(tensors)} tensors")
    
    def _find_input_for_output(self, output_idx: int) -> int:
        """Find which input connects to given output."""
        for input_idx, outputs in self.connectivity.items():
            if output_idx in outputs:
                return input_idx
        # Fallback - shouldn't happen with proper connectivity
        return output_idx % self.config.input_dim
    
    @debug_timer
    @debug_trace
    def apply(self, input_mps):
        """
        Apply expansion to input MPS.
        
        The key insight: We contract the input MPS with our expansion
        network along the upper physical indices, producing an expanded
        output MPS.
        """
        self._application_count += 1
        
        if self.debug:
            print(f"\n[APPLY #{self._application_count}] ExpansionOperator")
            print(f"  Input MPS: L={len(input_mps.tensors)}")
        
        # Copy input to avoid modifying original
        input_copy = input_mps.copy()
        expansion_copy = self.tn.copy()
        
        # Reindex input MPS to match our upper indices
        for i, tensor in enumerate(input_copy.tensors):
            # Replace the physical index with our naming convention
            old_ind = f'k{i}'
            if old_ind in tensor.inds:
                tensor.reindex_({old_ind: f'k{i}'})
            else:
                # Find the physical index (usually the smallest dimension)
                phys_idx = min(tensor.inds, key=lambda x: tensor.ind_size(x))
                tensor.reindex_({phys_idx: f'k{i}'})
        
        # Contract input with expansion network
        result = input_copy | expansion_copy
        
        # Contract all k indices
        for i in range(self.config.input_dim):
            k_ind = f'k{i}'
            if k_ind in result.ind_map:
                result.contract_ind(k_ind)
        
        # The result should now be tensors with:
        # - Virtual bonds between adjacent positions
        # - Lower physical indices (b) for outputs
        
        # Convert to MPS format
        output_arrays = []
        for i in range(self.config.output_dim):
            # Find tensor with tag O{i}
            tensor = result[f'O{i}']
            
            # Reshape to standard MPS format (left, right, physical)
            # Current shape might be (left, right, physical)
            data = tensor.data
            
            # Ensure correct shape
            if len(data.shape) == 4:
                # Squeeze out size-1 dimensions
                data = jnp.squeeze(data)
            
            if len(data.shape) == 3:
                # Already in correct format
                output_arrays.append(data)
            elif len(data.shape) == 2:
                # Add missing dimension
                if i == 0 or i == self.config.output_dim - 1:
                    # Edge tensor - add virtual dimension
                    data = jnp.expand_dims(data, axis=0 if i == 0 else 1)
                else:
                    # Shouldn't happen for middle tensors
                    data = jnp.expand_dims(data, axis=-1)
                output_arrays.append(data)
            else:
                # Fallback
                target_shape = self._get_mps_shape(i)
                data = jnp.reshape(data, target_shape)
                output_arrays.append(data)
        
        # Create output MPS
        output_mps = MatrixProductState(output_arrays, shape='lrp')
        
        if self.debug:
            print(f"  Output MPS: L={len(output_mps.tensors)}")
            print(f"  Output norm: {output_mps.norm():.6f}")
        
        return output_mps
    
    def _get_mps_shape(self, position: int) -> tuple:
        """Get expected MPS tensor shape at position."""
        if position == 0:
            return (1, self.config.bond_dim, self.config.phys_dim[1])
        elif position == self.config.output_dim - 1:
            return (self.config.bond_dim, 1, self.config.phys_dim[1])
        else:
            return (self.config.bond_dim, self.config.bond_dim, self.config.phys_dim[1])
    
    def get_config(self) -> LayerConfig:
        """Return configuration."""
        return self.config
    
    @property
    def tensors(self):
        """Access to underlying tensors."""
        return self.tn.tensors
    
    @property
    def implementation(self):
        """For unified operator compatibility."""
        return self.tn
    
    def __repr__(self):
        """String representation."""
        return (f"ExpansionOperator({self.config.input_dim}→"
                f"{self.config.output_dim}, χ={self.config.bond_dim})")