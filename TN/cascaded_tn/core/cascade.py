"""
Cascade container for managing sequences of operators.

This module provides the main TensorNetworkCascade class that manages
multiple cascadable operators as a unified network.
"""

from typing import List, Optional, Union, Any, Tuple
import time
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

from .base import (
    CascadableOperator, DebugInfo, CascadeValidationError,
    DimensionMismatchError, debug_timer, debug_trace
)
from .operator import CascadableSMPO


class TensorNetworkCascade:
    """
    Container for cascaded tensor network operators.
    
    Features:
    - Automatic validation of layer compatibility
    - Slice notation for accessing sublayers
    - Partial application support
    - Comprehensive debugging
    - Support for both training and inference modes
    """
    
    def __init__(self, 
                 operators: List[CascadableOperator],
                 name: Optional[str] = None,
                 debug: bool = False,
                 validate: bool = True):
        """
        Initialize cascade with list of operators.
        
        Args:
            operators: List of cascadable operators in order
            name: Optional name for this cascade
            debug: Enable debug output
            validate: Validate cascade on construction
        """
        if not operators:
            raise ValueError("Cannot create empty cascade")
        
        self.operators = operators
        self.name = name or "Cascade"
        self.debug = debug
        self.debug_info = DebugInfo()
        
        # Track applications
        self._application_count = 0
        self._last_execution_time = None
        
        if self.debug:
            print(f"\n[CASCADE] Creating {self.name} with {len(operators)} layers")
            for i, op in enumerate(operators):
                print(f"  Layer {i}: {op}")
        
        # Validate cascade structure
        if validate:
            self._validate_cascade()
    
    def _validate_cascade(self):
        """Validate that all operators can connect properly."""
        issues = []
        
        for i in range(len(self.operators) - 1):
            curr = self.operators[i]
            next_op = self.operators[i + 1]
            
            valid, msg = curr.validates_with(next_op)
            if not valid:
                issues.append(f"Layer {i}→{i+1}: {msg}")
        
        if issues:
            self.debug_info.validation_errors = issues
            error_msg = f"Cascade validation failed:\n" + "\n".join(issues)
            raise CascadeValidationError(error_msg)
        
        if self.debug:
            print(f"[CASCADE] ✅ Validation passed")
    
    @debug_timer
    def apply(self, input_tn, store_intermediates: bool = False) -> Any:
        """
        Apply the cascade to an input tensor network.
        
        Args:
            input_tn: Input tensor network (e.g., MPS)
            store_intermediates: Whether to store intermediate results
            
        Returns:
            Output tensor network after all operators
        """
        self._application_count += 1
        start_time = time.time()
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"CASCADE FORWARD PASS #{self._application_count} - {self.name}")
            print(f"{'='*60}")
        
        current = input_tn
        intermediates = [current] if store_intermediates else []
        
        for i, op in enumerate(self.operators):
            layer_start = time.time()
            
            if self.debug:
                print(f"\nLayer {i}: {op}")
            
            try:
                # Apply operator
                current = op.apply(current)

                # Apply ReLU if configured for this layer
                if op.config.enable_relu:
                    for tensor in current.tensors:
                        tensor.modify(data=jnp.maximum(0, tensor.data))
                    if self.debug:
                        print(f"  Applied ReLU activation")
                
                if self.debug:
                    # Record timing and info
                    layer_time = time.time() - layer_start
                    layer_norm = current.norm() if hasattr(current, 'norm') else 0.0
                    layer_shape = self._get_shape(current)
                    
                    self.debug_info.add_layer_info(i, layer_time, layer_norm, layer_shape)
                
                if store_intermediates:
                    intermediates.append(current)
                    
            except Exception as e:
                print(f"\n[ERROR] Layer {i} failed: {str(e)}")
                self.debug_info.validation_errors.append(f"Layer {i}: {str(e)}")
                raise
        
        self._last_execution_time = time.time() - start_time
        
        if self.debug:
            print(f"\n[CASCADE] Complete in {self._last_execution_time:.3f}s")
            print(f"{'='*60}\n")
        
        if store_intermediates:
            return current, intermediates
        return current
    
    def apply_partial(self, input_tn, start: int = 0, 
                     end: Optional[int] = None) -> Any:
        """
        Apply only a subset of layers.
        
        Useful for:
        - Extracting encoder output in autoencoders
        - Layer-wise analysis
        - Debugging specific layers
        """
        end = end or len(self.operators)
        
        if self.debug:
            print(f"\n[PARTIAL] Applying layers {start}:{end}")
        
        # Create temporary cascade with subset
        partial_cascade = TensorNetworkCascade(
            self.operators[start:end],
            name=f"{self.name}[{start}:{end}]",
            debug=self.debug,
            validate=True
        )
        
        return partial_cascade.apply(input_tn)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[CascadableOperator, 'TensorNetworkCascade']:
        """
        Slice notation support.
        
        cascade[0] -> first operator
        cascade[0:2] -> new cascade with first two operators
        cascade.encoder -> first half (property)
        """
        if isinstance(key, int):
            return self.operators[key]
        elif isinstance(key, slice):
            return TensorNetworkCascade(
                self.operators[key],
                name=f"{self.name}{key}",
                debug=self.debug,
                validate=True
            )
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __len__(self):
        """Number of operators in cascade."""
        return len(self.operators)
    
    @property
    def encoder(self) -> 'TensorNetworkCascade':
        """First half of cascade (for autoencoders)."""
        mid = len(self.operators) // 2
        return self[:mid]
    
    @property
    def decoder(self) -> 'TensorNetworkCascade':
        """Second half of cascade (for autoencoders)."""
        mid = len(self.operators) // 2
        return self[mid:]
    
    @property
    def bottleneck_index(self) -> int:
        """Index of the bottleneck layer (smallest dimension)."""
        dims = [op.output_dim for op in self.operators]
        # Add input dimension of first operator
        dims = [self.operators[0].input_dim] + dims
        return np.argmin(dims)
    
    def get_layer_info(self, index: int) -> dict:
        """Get detailed information about a specific layer."""
        if index >= len(self.operators):
            raise IndexError(f"Layer {index} out of range")
        
        op = self.operators[index]
        info = {
            'operator': str(op),
            'input_dim': op.input_dim,
            'output_dim': op.output_dim,
            'cyclic': op.is_cyclic,
            'config': op.get_config()
        }
        
        # Add debug info if available
        if hasattr(op, 'get_debug_info'):
            info['debug'] = op.get_debug_info()
        
        return info
    
    def summary(self):
        """Print a summary of the cascade architecture."""
        print(f"\n{'='*60}")
        print(f"CASCADE SUMMARY: {self.name}")
        print(f"{'='*60}")
        
        # Architecture overview
        dims = [self.operators[0].input_dim]
        dims.extend([op.output_dim for op in self.operators])
        arch_str = " → ".join(str(d) for d in dims)
        print(f"Architecture: {arch_str}")
        
        # Layer details
        print(f"\nLayers: {len(self.operators)}")
        for i, op in enumerate(self.operators):
            marker = " (bottleneck)" if i == self.bottleneck_index - 1 else ""
            print(f"  {i}: {op}{marker}")
        
        # Execution stats
        if self._application_count > 0:
            print(f"\nExecution Stats:")
            print(f"  Applications: {self._application_count}")
            if self._last_execution_time:
                print(f"  Last execution: {self._last_execution_time:.3f}s")
        
        # Debug info
        if self.debug_info.layer_timings:
            avg_time = np.mean(self.debug_info.layer_timings)
            print(f"  Avg layer time: {avg_time:.3f}s")
        
        print(f"{'='*60}\n")
    
    def _get_shape(self, tn) -> Any:
        """Extract shape information from tensor network."""
        if hasattr(tn, 'shape'):
            return tn.shape
        elif hasattr(tn, 'L'):
            return f"L={tn.L}"
        else:
            return "unknown"
    
    def prepare_for_training(self) -> qtn.TensorNetwork:
        """
        Prepare cascade for tn4ml training by creating connected TensorNetwork.
        
        This method handles the complex index renaming needed to connect
        SMPO layers into a single trainable network.
        """
        if self.debug:
            print(f"\n[TRAINING] Preparing cascade for training...")
        
        # For now, return a placeholder
        # Full implementation will handle index connections
        print("[WARNING] prepare_for_training not yet fully implemented")
        
        # Collect all tensors
        all_tensors = []
        for i, op in enumerate(self.operators):
            if hasattr(op, 'smpo') and hasattr(op.smpo, 'tensors'):
                all_tensors.extend(op.smpo.tensors)
        
        return qtn.TensorNetwork(all_tensors)
    
    @property
    def L(self) -> int:
        """Total number of tensors (for Model compatibility)."""
        return sum(op.config.input_dim for op in self.operators)
    
    @property
    def tensors(self) -> List[Any]:
        """All tensors from all operators (for loss function compatibility)."""
        all_tensors = []
        for op in self.operators:
            if hasattr(op, 'smpo') and hasattr(op.smpo, 'tensors'):
                all_tensors.extend(op.smpo.tensors)
        return all_tensors
    
    def __repr__(self):
        """String representation."""
        dims = [self.operators[0].input_dim]
        dims.extend([op.output_dim for op in self.operators])
        arch = "→".join(str(d) for d in dims)
        return f"TensorNetworkCascade({arch}, layers={len(self.operators)})"

    def copy(self):
        """
        Create a copy of the cascade.
        
        Returns:
            TensorNetworkCascade: A new cascade with copied operators
        """
        # Create copies of all operators
        copied_operators = []
        
        for op in self.operators:
            # For UnifiedCascadableOperator
            if hasattr(op, 'implementation'):
                # Create a new operator with the same config
                new_op = type(op)(
                    config=op.config,
                    debug=op.debug,
                    debug_level=op.debug_level
                )
                
                # Copy the tensor data
                if hasattr(op.implementation, 'tensors'):
                    for old_t, new_t in zip(op.implementation.tensors, new_op.implementation.tensors):
                        new_t.modify(data=old_t.data.copy())
                
                copied_operators.append(new_op)
            else:
                # Fallback for other operator types
                # This is a shallow copy - might need to be more sophisticated
                import copy
                copied_operators.append(copy.deepcopy(op))
        
        # Create new cascade with copied operators
        new_cascade = TensorNetworkCascade(
            operators=copied_operators,
            name=self.name + "_copy",
            debug=self.debug,
            validate=False  # Skip validation since we know it's valid
        )
        
        return new_cascade