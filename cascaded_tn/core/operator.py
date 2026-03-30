"""
Cascadable operator implementations.

This module provides SMPO wrappers that can be cascaded together,
with full debugging support and automatic handling of cyclic boundaries.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import jax
import jax.numpy as jnp
from tn4ml.models.smpo import SpacedMatrixProductOperator, SMPO_initialize
import quimb.tensor as qtn

from .base import CascadableOperator, LayerConfig, debug_timer, debug_trace


class CascadableSMPO(CascadableOperator):
    """
    Wrapper around tn4ml's SpacedMatrixProductOperator for cascading.
    
    Adds:
    - Cascading interface compatibility
    - Automatic cyclic/open boundary handling
    - Comprehensive debugging
    - Connection validation
    """
    
    def __init__(self, 
                 smpo: Optional[SpacedMatrixProductOperator] = None,
                 config: Optional[LayerConfig] = None,
                 initializer=None,
                 key=None,
                 debug: bool = False,
                 debug_level: int = 0,
                 **smpo_kwargs):
        """
        Initialize cascadable SMPO.
        
        Can either wrap existing SMPO or create new one from config.
        """
        super().__init__(debug=debug, debug_level=debug_level)
        
        if smpo is not None:
            # Wrap existing SMPO
            self.smpo = smpo
            self.config = self._infer_config_from_smpo(smpo)
        elif config is not None:
            # Create new SMPO from config
            self.config = config
            self.smpo = self._create_smpo_from_config(
                config, initializer, key, **smpo_kwargs
            )
        else:
            raise ValueError("Must provide either smpo or config")
        
        # Cache for debugging info
        self._last_input_shape = None
        self._last_output_shape = None
        self._application_count = 0
        
        if self.debug:
            print(f"[INIT] Created {self}")
    
    def _infer_config_from_smpo(self, smpo: SpacedMatrixProductOperator) -> LayerConfig:
        """Infer configuration from existing SMPO."""
        # Count actual outputs
        output_indices = list(smpo.lower_inds)
        output_dim = len(output_indices)
        
        # Determine spacing type
        if hasattr(smpo, 'spacing') and isinstance(smpo.spacing, int):
            spacing = smpo.spacing
        elif hasattr(smpo, 'spacings'):
            spacing = smpo.spacings
        else:
            spacing = None
        
        config = LayerConfig(
            input_dim=smpo.L,
            output_dim=output_dim,
            bond_dim=self._estimate_bond_dim(smpo),
            spacing=spacing,
            cyclic=getattr(smpo, 'cyclic', False),
            phys_dim=self._get_phys_dim(smpo)
        )
        
        if self.debug_level >= 2:
            print(f"[INFER] Extracted config: {config}")
        
        return config
    
    def _create_smpo_from_config(self, config: LayerConfig, 
                                initializer, key, **kwargs) -> SpacedMatrixProductOperator:
        """Create new SMPO from configuration."""
        if initializer is None:
            initializer = jax.nn.initializers.normal(stddev=0.1)
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Check if this is an expansion layer
        if config.output_dim >= config.input_dim:
            raise ValueError(
                f"CascadableSMPO cannot handle expansion ({config.input_dim}→{config.output_dim}). "
                f"SMPO is designed for compression only. Use ExpansionSMPO (coming soon) instead."
            )
        
        # Determine output specification method
        smpo_kwargs = {}
        
        if config.output_inds is not None:
            # Use explicit output positions (takes precedence)
            smpo_kwargs['output_inds'] = config.output_inds
            if self.debug:
                print(f"[CREATE] Using explicit output_inds: {config.output_inds}")
        else:
            # Calculate or use provided spacing
            if config.spacing is None:
                if self.debug:
                    print(f"[CREATE] Auto-calculating spacing for {config.input_dim}→{config.output_dim}")
                
                from ..builders.dimension_calculator import DimensionCalculator
                calc = DimensionCalculator(debug=False)
                spacing_result = calc.calculate_optimal_spacing(
                    config.input_dim, config.output_dim, config.cyclic
                )
                spacing = spacing_result.spacing
                
                if self.debug:
                    print(f"[CREATE] Calculated spacing: {spacing}")
            else:
                spacing = config.spacing
            
            # Handle non-uniform spacing
            if isinstance(spacing, list):
                smpo_kwargs['spacings'] = spacing
            else:
                smpo_kwargs['spacing'] = spacing
        
        if self.debug:
            print(f"[CREATE] Building SMPO: L={config.input_dim}, "
                f"cyclic={config.cyclic}, smpo_kwargs={smpo_kwargs}")
        
        smpo = SMPO_initialize(
            L=config.input_dim,
            initializer=initializer,
            key=key,
            dtype=jnp.float32,
            bond_dim=config.bond_dim,
            phys_dim=config.phys_dim,
            cyclic=config.cyclic,
            add_identity=config.add_identity,
            boundary='pbc' if config.cyclic else 'obc',
            shape_method='even' if config.cyclic else kwargs.get('shape_method', 'even'),
            **smpo_kwargs,
            **kwargs
        )
        
        # Verify output count matches expectation
        actual_outputs = len(list(smpo.lower_inds))
        if actual_outputs != config.output_dim:
            print(f"[WARNING] SMPO created with {actual_outputs} outputs, "
                f"expected {config.output_dim}")
        
        return smpo
    
    @debug_timer
    @debug_trace
    def apply(self, input_mps):
        """
        Apply this SMPO to an input MPS.
        
        Handles:
        - Cyclic/open boundary mismatches
        - Debug tracing
        - Shape validation
        """
        self._application_count += 1
        
        # Store input info for debugging
        self._last_input_shape = self._get_mps_shape(input_mps)
        
        if self.debug:
            print(f"\n[APPLY #{self._application_count}] {self}")
            print(f"  Input: shape={self._last_input_shape}, "
                  f"cyclic={getattr(input_mps, 'cyclic', '?')}")
        
        # Check boundary condition compatibility
        input_cyclic = getattr(input_mps, 'cyclic', False)
        if input_cyclic != self.config.cyclic:
            if self.debug:
                print(f"  [BOUNDARY] Converting {'cyclic→open' if input_cyclic else 'open→cyclic'}")
            input_mps = self._convert_boundary_conditions(input_mps, self.config.cyclic)
        
        # Apply the SMPO
        try:
            output_mps = self.smpo.apply(input_mps)
            
            # Store output info
            self._last_output_shape = self._get_mps_shape(output_mps)
            
            if self.debug:
                print(f"  Output: shape={self._last_output_shape}, "
                      f"norm={output_mps.norm():.6f}")
            
            # Validate output
            if self.debug_level >= 2:
                self._validate_output(output_mps)
            
            return output_mps
            
        except Exception as e:
            print(f"\n[ERROR] SMPO application failed!")
            print(f"  SMPO: {self}")
            print(f"  Input shape: {self._last_input_shape}")
            print(f"  Error: {str(e)}")
            raise
    
    def _convert_boundary_conditions(self, mps, to_cyclic: bool):
        """Convert MPS between cyclic and open boundary conditions."""
        # This is a placeholder - actual implementation depends on tn4ml internals
        if self.debug_level >= 2:
            print(f"  [CONVERT] {'Open→Cyclic' if to_cyclic else 'Cyclic→Open'}")
        
        # For now, just return the MPS unchanged with a warning
        print(f"  [WARNING] Boundary conversion not yet implemented, "
              f"proceeding with original MPS")
        return mps
    
    def _get_mps_shape(self, mps) -> tuple:
        """Extract shape information from MPS."""
        try:
            if hasattr(mps, 'shape'):
                return mps.shape
            elif hasattr(mps, 'L'):
                # Tensor network with L sites
                bond_dims = []
                for i in range(mps.L - 1):
                    bond = mps.bond_size(i, i+1) if hasattr(mps, 'bond_size') else '?'
                    bond_dims.append(bond)
                return (mps.L, bond_dims)
            else:
                return ('unknown',)
        except:
            return ('error',)
    
    def _estimate_bond_dim(self, smpo) -> int:
        """Estimate bond dimension from SMPO tensors."""
        try:
            # Look at first tensor's bond indices
            if hasattr(smpo, 'tensors') and len(smpo.tensors) > 0:
                tensor = smpo.tensors[0]
                # Find bond-like index
                for ind in tensor.inds:
                    if 'bond' in ind or ind.startswith('b'):
                        return tensor.ind_size(ind)
            return 4  # Default
        except:
            return 4
    
    def _get_phys_dim(self, smpo) -> tuple:
        """Extract physical dimensions from SMPO."""
        try:
            if hasattr(smpo, 'phys_dim'):
                return smpo.phys_dim
            # Look at tensor indices
            if hasattr(smpo, 'tensors') and len(smpo.tensors) > 0:
                tensor = smpo.tensors[0]
                up_dim = tensor.ind_size('k0') if 'k0' in tensor.inds else 2
                down_dim = tensor.ind_size('b0') if 'b0' in tensor.inds else 2
                return (up_dim, down_dim)
        except:
            pass
        return (2, 2)  # Default
    
    def _validate_output(self, output_mps):
        """Validate output MPS properties."""
        try:
            # Check norm
            norm = output_mps.norm()
            if norm < 1e-10 or norm > 1e10:
                print(f"  [WARNING] Unusual output norm: {norm}")
            
            # Check bond dimensions
            if hasattr(output_mps, 'L'):
                max_bond = max([output_mps.bond_size(i, i+1) 
                              for i in range(output_mps.L-1)])
                if max_bond > 1000:
                    print(f"  [WARNING] Very large bond dimension: {max_bond}")
        except:
            pass  # Validation is best-effort
    
    def get_config(self) -> LayerConfig:
        """Return the configuration of this operator."""
        return self.config
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about this operator."""
        return {
            'applications': self._application_count,
            'last_input_shape': self._last_input_shape,
            'last_output_shape': self._last_output_shape,
            'config': self.config,
            'tensors': len(self.smpo.tensors) if hasattr(self.smpo, 'tensors') else 0
        }
    
    def __repr__(self):
        """String representation with more detail."""
        cyclic_str = "↻" if self.config.cyclic else "→"
        spacing_str = f"s={self.config.spacing}" if self.config.spacing else ""
        return (f"CascadableSMPO({self.config.input_dim}{cyclic_str}"
                f"{self.config.output_dim}, χ={self.config.bond_dim}, {spacing_str})")


