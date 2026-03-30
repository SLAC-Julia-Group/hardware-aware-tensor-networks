"""
Base classes and interfaces for cascaded tensor networks.

This module provides the foundation for building cascaded tensor network
architectures with comprehensive debugging support.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from functools import wraps
import time


def debug_timer(func):
    """Decorator to time function execution when debug is enabled."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'debug', False):
            return func(self, *args, **kwargs)
            
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start
        
        print(f"[TIMER] {self.__class__.__name__}.{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def debug_trace(func):
    """Decorator to trace function calls with input/output info."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'debug_level', 0) >= 2:
            return func(self, *args, **kwargs)
            
        print(f"\n[TRACE] Entering {self.__class__.__name__}.{func.__name__}")
        print(f"        Args: {[type(a).__name__ for a in args]}")
        print(f"        Kwargs: {list(kwargs.keys())}")
        
        result = func(self, *args, **kwargs)
        
        print(f"[TRACE] Exiting {self.__class__.__name__}.{func.__name__}")
        if hasattr(result, 'shape'):
            print(f"        Result shape: {result.shape}")
        
        return result
    return wrapper


@dataclass
class LayerConfig:
    """Configuration for a single layer in the cascade."""
    input_dim: int
    output_dim: int
    bond_dim: int
    spacing: Optional[Union[int, List[int]]] = None  # Auto-calculated if None
    output_inds: Optional[List[int]] = None  # Arbitrary output positions (overrides spacing)
    cyclic: bool = False
    phys_dim: Tuple[int, int] = (2, 2)
    add_identity: bool = False
    enable_relu: bool = False  # Apply ReLU activation after this layer
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError(f"Dimensions must be positive: input={self.input_dim}, output={self.output_dim}")
        
        if self.output_dim > self.input_dim:
            raise ValueError(
                f"Expansion layers are not supported: {self.input_dim} → {self.output_dim}. "
                f"Each layer must compress or preserve dimension (output_dim <= input_dim)."
            )
        
        if self.bond_dim < 2:
            raise ValueError(f"Bond dimension must be at least 2, got {self.bond_dim}")

        # Validate output_inds if provided
        if self.output_inds is not None:
            if len(self.output_inds) != self.output_dim:
                raise ValueError(
                    f"output_inds length ({len(self.output_inds)}) must match "
                    f"output_dim ({self.output_dim})"
                )
            if self.output_inds != sorted(self.output_inds):
                raise ValueError(f"output_inds must be sorted, got {self.output_inds}")
            if len(self.output_inds) != len(set(self.output_inds)):
                raise ValueError(f"output_inds must be unique, got {self.output_inds}")
            if any(idx < 0 or idx >= self.input_dim for idx in self.output_inds):
                raise ValueError(
                    f"output_inds must be in range [0, {self.input_dim-1}], "
                    f"got {self.output_inds}"
                )
            # output_inds takes precedence over spacing
            if self.spacing is not None:
                print(f"[INFO] output_inds provided, ignoring spacing={self.spacing}")
                self.spacing = None

class CascadableOperator(ABC):
    """
    Abstract base class for operators that can be cascaded.
    
    This provides the interface that all cascadable tensor network
    operators must implement.
    """
    
    def __init__(self, debug: bool = False, debug_level: int = 0):
        self.debug = debug
        self.debug_level = debug_level
        self._call_count = 0
    
    @abstractmethod
    def apply(self, input_tn):
        """Apply this operator to an input tensor network."""
        pass
    
    @abstractmethod
    def get_config(self) -> LayerConfig:
        """Return the configuration of this operator."""
        pass
    
    @property
    def input_dim(self) -> int:
        """Number of input indices expected."""
        return self.get_config().input_dim
    
    @property
    def output_dim(self) -> int:
        """Number of output indices produced."""
        return self.get_config().output_dim
    
    @property
    def is_cyclic(self) -> bool:
        """Whether this operator has cyclic boundary conditions."""
        return self.get_config().cyclic
    
    def validates_with(self, other: 'CascadableOperator') -> Tuple[bool, str]:
        """
        Check if this operator can connect to another.
        
        Returns:
            (is_valid, error_message)
        """
        if self.output_dim != other.input_dim:
            return False, f"Dimension mismatch: {self.output_dim} outputs != {other.input_dim} inputs"
        
        if self.is_cyclic != other.is_cyclic:
            # This is a warning, not an error - we can handle it
            msg = f"Boundary condition mismatch: {'cyclic' if self.is_cyclic else 'open'} → {'cyclic' if other.is_cyclic else 'open'}"
            print(f"[WARNING] {msg}")
        
        return True, ""
    
    def __repr__(self):
        config = self.get_config()
        cyclic_str = "↻" if config.cyclic else "→"
        return f"{self.__class__.__name__}({config.input_dim}{cyclic_str}{config.output_dim}, χ={config.bond_dim})"


class DebugInfo:
    """Container for debugging information during cascade operations."""
    
    def __init__(self):
        self.layer_timings: List[float] = []
        self.layer_norms: List[float] = []
        self.layer_shapes: List[Any] = []
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_layer_info(self, layer_idx: int, timing: float, norm: float, shape: Any):
        """Record information about a layer's execution."""
        self.layer_timings.append(timing)
        self.layer_norms.append(norm)
        self.layer_shapes.append(shape)
        
        if self.layer_timings:
            # Handle JAX tracers during gradient computation
            try:
                print(f"[DEBUG] Layer {layer_idx}: {timing:.3f}s, norm={norm:.6f}, shape={shape}")
            except TypeError:
                # During tracing, just print without formatting
                print(f"[DEBUG] Layer {layer_idx}: {timing}s, norm={norm}, shape={shape}")
    
    def summary(self):
        """Print a summary of the debugging information."""
        print("\n" + "="*60)
        print("CASCADE EXECUTION SUMMARY")
        print("="*60)
        
        if self.layer_timings:
            print(f"Total layers: {len(self.layer_timings)}")
            print(f"Total time: {sum(self.layer_timings):.3f}s")
            print(f"Average time per layer: {np.mean(self.layer_timings):.3f}s")
            print(f"Slowest layer: {np.argmax(self.layer_timings)} ({max(self.layer_timings):.3f}s)")
        
        if self.validation_errors:
            print(f"\n⚠️  Validation Errors: {len(self.validation_errors)}")
            for err in self.validation_errors:
                print(f"   - {err}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warn in self.warnings:
                print(f"   - {warn}")
        
        print("="*60 + "\n")


class CascadeValidationError(Exception):
    """Raised when cascade validation fails."""
    pass


class DimensionMismatchError(Exception):
    """Raised when layer dimensions don't match."""
    pass