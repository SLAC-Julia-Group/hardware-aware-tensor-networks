"""
expansion.py - Expansion operator for cascaded tensor networks.

This module provides expansion operations (few inputs → many outputs)
using the ExpansionMPO implementation.
"""

from typing import Dict, List, Optional, Any
import jax
import jax.numpy as jnp

from .base import CascadableOperator, LayerConfig, debug_timer, debug_trace
from .expansion_mpo import ExpansionMPO, expansion_mpo_initialize


class ExpansionOperator(CascadableOperator):
    """
    Expansion operator that uses ExpansionMPO internally.
    
    This provides a clean interface for the cascade architecture
    while using the robust ExpansionMPO implementation.
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
        self._application_count = 0
        
        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Calculate which positions should have inputs
        # Simple strategy: distribute inputs evenly
        num_inputs = config.input_dim
        num_outputs = config.output_dim
        
        # Calculate spacing
        spacing = max(1, num_outputs // num_inputs)
        
        # Generate input positions
        input_positions = []
        for i in range(num_inputs):
            pos = i * num_outputs // num_inputs
            input_positions.append(pos)
        
        if self.debug:
            print(f"[EXPANSION] Creating {config.input_dim}→{config.output_dim}")
            print(f"[EXPANSION] Input positions: {input_positions}")
        
        # Create ExpansionMPO
        self.expansion_mpo = expansion_mpo_initialize(
            L=config.output_dim,
            num_inputs=config.input_dim,
            input_positions=input_positions,
            initializer=initializer,
            key=key,
            bond_dim=config.bond_dim,
            phys_dim=config.phys_dim,
            cyclic=config.cyclic,
            debug=debug and debug_level >= 2,
            **kwargs
        )
        
        if self.debug:
            print(f"[EXPANSION] Created ExpansionMPO with {len(self.expansion_mpo.tensors)} tensors")
    
    @debug_timer
    @debug_trace
    def apply(self, input_mps):
        """Apply expansion to input MPS."""
        self._application_count += 1
        
        if self.debug:
            print(f"\n[APPLY #{self._application_count}] ExpansionOperator")
            print(f"  Input MPS: L={input_mps.L if hasattr(input_mps, 'L') else len(input_mps.tensors)}")
        
        # Use ExpansionMPO's apply method
        output_mps = self.expansion_mpo.apply(input_mps)
        
        if self.debug:
            print(f"  Output MPS: L={output_mps.L}")
            print(f"  Output norm: {output_mps.norm():.6f}")
        
        return output_mps
    
    def get_config(self) -> LayerConfig:
        """Return configuration."""
        return self.config
    
    @property
    def tensors(self):
        """Access to underlying tensors for training."""
        return self.expansion_mpo.tensors
    
    @property
    def implementation(self):
        """For unified operator compatibility."""
        return self.expansion_mpo
    
    @property
    def L(self):
        """Number of tensors."""
        return self.expansion_mpo.L
    
    @property
    def arrays(self):
        """Tensor arrays for training."""
        return [t.data for t in self.expansion_mpo.tensors]
    
    def update_tensors(self, arrays):
        """Update tensor data from arrays (for training)."""
        for tensor, array in zip(self.expansion_mpo.tensors, arrays):
            tensor.modify(data=array)
    
    def normalize(self, insert=None):
        """Normalize the operator."""
        self.expansion_mpo.normalize(insert=insert)
    
    def copy(self):
        """Create a copy of this operator."""
        new_op = ExpansionOperator(
            config=self.config,
            debug=self.debug,
            debug_level=self.debug_level
        )
        
        # Copy tensor data
        for old_t, new_t in zip(self.expansion_mpo.tensors, new_op.expansion_mpo.tensors):
            new_t.modify(data=old_t.data.copy())
        
        return new_op
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information."""
        debug_info = self.expansion_mpo.get_debug_info()
        debug_info.update({
            'applications': self._application_count,
            'config': self.config
        })
        return debug_info
    
    def __repr__(self):
        """String representation."""
        return (f"ExpansionOperator({self.config.input_dim}→"
                f"{self.config.output_dim}, χ={self.config.bond_dim})")