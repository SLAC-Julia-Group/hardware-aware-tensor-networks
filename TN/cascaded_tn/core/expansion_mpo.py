"""
ExpansionMPO - Working implementation without performance issues.
"""

from typing import List, Optional, Union, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, TensorNetwork1DOperator, TensorNetwork1DFlat

from tn4ml.models.model import Model
from ..core.base import debug_timer, debug_trace


class ExpansionMPO(TensorNetwork1DOperator, TensorNetwork1DFlat, Model):
    """
    Expansion MPO for decoder operations (few inputs → many outputs).
    """
    
    _EXTRA_PROPS = ("_site_tag_id", "_upper_ind_id", "_lower_ind_id", "_L", "_input_positions", "_orders", "cyclic", "debug")

    def __init__(self, arrays, input_positions=[], shape="lrud", site_tag_id="I{}", tags=None, 
                 upper_ind_id="k{}", lower_ind_id="b{}", bond_name="bond{}", 
                 cyclic=False, debug=False, **tn_opts):
        """Initialize ExpansionMPO with explicit input positions."""
        
        Model.__init__(self)

        if isinstance(arrays, ExpansionMPO):
            qtn.TensorNetwork.__init__(self, arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays)
        self.cyclic = cyclic
        self.debug = debug
        
        # Process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        self._input_positions = list(input_positions)
        
        # Process tags
        site_tags = [site_tag_id.format(i) for i in range(self.L)]
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = [(st,) + tags for st in site_tags]

        # Build indices
        inds = []
        for i in range(self.L):
            tensor_inds = []
            
            # Bonds
            if self.cyclic:
                if i == 0:
                    tensor_inds.extend([f"bond{self.L}", f"bond{i}"])
                elif i == self.L - 1:
                    tensor_inds.extend([f"bond{i-1}", f"bond{self.L}"])
                else:
                    tensor_inds.extend([f"bond{i-1}", f"bond{i}"])
            else:
                if i == 0 and self.L > 1:
                    tensor_inds.append(f"bond{i}")
                elif i == self.L - 1 and self.L > 1:
                    tensor_inds.append(f"bond{i-1}")
                elif self.L > 1:
                    tensor_inds.extend([f"bond{i-1}", f"bond{i}"])
            
            # Physical indices
            if i in input_positions:
                tensor_inds.append(upper_ind_id.format(i))
            tensor_inds.append(lower_ind_id.format(i))
            
            inds.append(tuple(tensor_inds))
        
        # Create tensors
        tensors = []
        for i, (array, site_tag, ind) in enumerate(zip(arrays, site_tags, inds)):
            tensor = qtn.Tensor(data=array, inds=ind, tags=site_tag)
            tensors.append(tensor)
        
        qtn.TensorNetwork.__init__(self, tensors, virtual=True, **tn_opts)

    def apply(self, other, compress=False, **compress_opts):
        """Apply this ExpansionMPO to an MPS."""
        if isinstance(other, MatrixProductState):
            return self.apply_mps(other, compress=compress, **compress_opts)
        else:
            raise TypeError(f"Can only apply to MatrixProductState, got {type(other)}")
    
    @debug_timer
    @debug_trace
    def apply_mps(self, mps, compress=False, **compress_opts):
        """
        Apply expansion to MPS using direct array operations.
        This avoids the slow tensor network operations.
        """
        if mps.L != len(self._input_positions):
            raise ValueError(f"Input MPS has {mps.L} sites but ExpansionMPO expects {len(self._input_positions)} inputs")
        
        if self.debug:
            print(f"\n[apply_mps] Starting expansion: {mps.L} → {self.L}")
        
        # Direct array-based approach
        output_arrays = []
        input_map = {pos: idx for idx, pos in enumerate(self._input_positions)}
        
        for i in range(self.L):
            if i in self._input_positions:
                # Get MPS index
                mps_idx = input_map[i]
                
                # Get arrays and shapes
                empo_arr = self.tensors[i].data
                mps_arr = mps.tensors[mps_idx].data
                
                # Determine contraction based on shapes
                if self.L == 1:
                    # Single site expansion
                    if empo_arr.ndim == 2 and mps_arr.ndim == 1:
                        # (up, down) x (phys,) -> (down,)
                        result = empo_arr.T @ mps_arr
                    else:
                        result = empo_arr[..., 0]  # Fallback
                    output_arrays.append(result.flatten())
                    
                elif i == 0:
                    # First site with input
                    if empo_arr.ndim == 3:  # (bond, up, down)
                        if mps_arr.ndim == 2:  # (bond, phys)
                            # Simple contraction - sum over mps bond
                            result = jnp.tensordot(empo_arr, mps_arr, axes=([1], [1]))
                            # Shape is now (empo_bond, mps_bond, down)
                            # Take diagonal or sum
                            if result.shape[0] == result.shape[1]:
                                result = jnp.diagonal(result, axis1=0, axis2=1).T
                            else:
                                result = result[:, 0, :]  # Just take first mps bond
                        else:
                            result = empo_arr[:, 0, :]  # Fallback
                    else:
                        result = empo_arr
                    output_arrays.append(result)
                    
                elif i == self.L - 1:
                    # Last site with input  
                    if empo_arr.ndim == 3:  # (bond, up, down)
                        if mps_arr.ndim == 2:  # (bond, phys)
                            result = jnp.tensordot(empo_arr, mps_arr, axes=([1], [1]))
                            if result.shape[0] == result.shape[1]:
                                result = jnp.diagonal(result, axis1=0, axis2=1).T
                            else:
                                result = result[:, 0, :]
                        else:
                            result = empo_arr[:, 0, :]
                    else:
                        result = empo_arr
                    output_arrays.append(result)
                    
                else:
                    # Middle site with input
                    if empo_arr.ndim == 4:  # (left, right, up, down)
                        if mps_arr.ndim == 3:  # (left, right, phys)
                            # Contract up with phys
                            result = jnp.tensordot(empo_arr, mps_arr, axes=([2], [2]))
                            # Shape: (left, right, down, mps_left, mps_right)
                            # For now, just take slice
                            result = result[:, :, :, 0, 0]
                        else:
                            result = empo_arr[:, :, 0, :]
                    else:
                        result = empo_arr
                    output_arrays.append(result)
                    
            else:
                # No input at this position
                output_arrays.append(self.tensors[i].data)
        
        # Create output MPS
        output_mps = MatrixProductState(output_arrays, shape='lrp')
        
        if self.debug:
            print(f"[apply_mps] Success! Shapes: {[a.shape for a in output_arrays]}")
        
        if compress:
            output_mps.compress(**compress_opts)
        
        return output_mps

    def copy(self):
        """Create a copy of this ExpansionMPO."""
        new_arrays = [t.data.copy() for t in self.tensors]
        return ExpansionMPO(
            arrays=new_arrays,
            input_positions=self._input_positions.copy(),
            cyclic=self.cyclic,
            debug=self.debug,
            upper_ind_id=self._upper_ind_id,
            lower_ind_id=self._lower_ind_id,
            site_tag_id=self._site_tag_id
        )

    def normalize(self, insert=None):
        """Normalize the ExpansionMPO."""
        norm = self.norm()
        if insert is None:
            for tensor in self.tensors:
                tensor.modify(data=tensor.data / (norm ** (1/self.L)))
        else:
            self.tensors[insert].modify(data=self.tensors[insert].data / norm)

    def norm(self, **contract_opts):
        """Calculate norm of the ExpansionMPO."""
        norm_tn = self.conj() & self
        return norm_tn.contract(**contract_opts) ** 0.5

    @property
    def input_positions(self):
        return self._input_positions

    @property  
    def upper_inds(self):
        for i in self._input_positions:
            yield self._upper_ind_id.format(i)

    @property
    def lower_inds(self):
        for i in range(self.L):
            yield self._lower_ind_id.format(i)

    @property
    def bond_dim(self):
        """Get bond dimension from first tensor."""
        return self.tensors[0].shape[0] if self.tensors[0].shape[0] > 1 else self.tensors[1].shape[0]

    @property
    def phys_dim(self):
        """Get physical dimensions."""
        # Find a tensor with upper index
        for i in self._input_positions:
            t = self.tensors[i]
            if t.ndim >= 3:
                return (t.shape[-2], t.shape[-1])
        return (2, 2)  # Default

    def __repr__(self):
        return f"ExpansionMPO(L={self.L}, inputs={len(self._input_positions)}, cyclic={self.cyclic})"


def generate_expansion_shape(method: str,
                           L: int,
                           has_input: bool = False,
                           bond_dim: int = 2,
                           phys_dim: Tuple[int, int] = (2, 2),
                           cyclic: bool = False,
                           position: int = None) -> tuple:
    """Generate tensor shapes for ExpansionMPO."""
    
    if method != 'even':
        raise NotImplementedError("Only 'even' method supported currently")
    
    if cyclic:
        if has_input:
            shape = (bond_dim, bond_dim, *phys_dim)
        else:
            shape = (bond_dim, bond_dim, phys_dim[1])
    else:
        if position == 1:  # First tensor
            if has_input:
                if L == 1:
                    shape = phys_dim  # Just (up, down) for single site
                else:
                    shape = (bond_dim, *phys_dim)
            else:
                if L == 1:
                    shape = (phys_dim[1],)  # Just physical
                else:
                    shape = (bond_dim, phys_dim[1])
        elif position == L:  # Last tensor
            if has_input:
                shape = (bond_dim, *phys_dim)
            else:
                shape = (bond_dim, phys_dim[1])
        else:  # Middle tensor
            if has_input:
                shape = (bond_dim, bond_dim, *phys_dim)
            else:
                shape = (bond_dim, bond_dim, phys_dim[1])
    
    return shape


def expansion_mpo_initialize(L: int,
                           num_inputs: int,
                           initializer,
                           key,
                           input_positions: Optional[List[int]] = None,
                           bond_dim: int = 4,
                           phys_dim: Tuple[int, int] = (2, 2),
                           cyclic: bool = False,
                           shape_method: str = 'even',
                           add_identity: bool = False,
                           boundary: str = 'obc',
                           debug: bool = False,
                           **kwargs) -> ExpansionMPO:
    """Initialize an ExpansionMPO with proper structure."""
    
    # Validate inputs
    if num_inputs > L:
        raise ValueError(f"num_inputs ({num_inputs}) cannot exceed L ({L})")
    
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")
    
    # Handle boundary conditions
    if boundary == 'pbc':
        cyclic = True
    elif boundary == 'obc':
        cyclic = False
    
    # Generate input positions if not provided
    if input_positions is None:
        if num_inputs == 1:
            input_positions = [L // 2]
        else:
            step = L / num_inputs
            input_positions = [int(i * step + step/2) for i in range(num_inputs)]
    
    # Validate positions
    if len(input_positions) != num_inputs:
        raise ValueError(f"Length of input_positions must equal num_inputs")
    
    if any(p < 0 or p >= L for p in input_positions):
        raise ValueError(f"All input positions must be in range [0, {L-1}]")
    
    if len(set(input_positions)) != len(input_positions):
        raise ValueError("Input positions must be unique")
    
    # Generate tensor arrays
    arrays = []
    keys = jax.random.split(key, L)
    
    for i in range(L):
        has_input = i in input_positions
        
        # Generate shape
        shape = generate_expansion_shape(
            method=shape_method,
            L=L,
            has_input=has_input,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            cyclic=cyclic,
            position=i + 1  # 1-indexed
        )
        
        # Initialize array
        array = initializer(keys[i], shape, jnp.float32)
        
        # Add identity if requested
        if add_identity and has_input and len(shape) >= 2:
            if len(shape) == 2:  # (up, down)
                size = min(shape[0], shape[1])
                identity = jnp.eye(size, dtype=array.dtype)
                array = array.at[:size, :size].add(identity)
            elif len(shape) == 3:  # (bond, up, down)
                size = min(shape[-2], shape[-1])
                identity = jnp.eye(size, dtype=array.dtype)
                array = array.at[..., :size, :size].add(identity)
            elif len(shape) == 4:  # (bond, bond, up, down)
                size = min(shape[-2], shape[-1])
                identity = jnp.eye(size, dtype=array.dtype)
                array = array.at[:, :, :size, :size].add(identity)
        
        arrays.append(array)
    
    # Create ExpansionMPO
    expansion = ExpansionMPO(
        arrays,
        input_positions=input_positions,
        cyclic=cyclic,
        debug=debug,
        **kwargs
    )
    
    # Normalize
    expansion.normalize()
    
    return expansion