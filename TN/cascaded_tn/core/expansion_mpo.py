"""
expansion_mpo.py - Expansion MPO for decoder layers in cascaded tensor networks.

This module implements the reverse of SMPO - instead of many inputs to few outputs,
it handles few inputs to many outputs by reversing the spacing logic.
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
    An MPO for expansion operations - the reverse of SMPO.
    
    Instead of having many inputs with few outputs (compression),
    this has few inputs with many outputs (expansion).
    
    Key differences from SMPO:
    - Only some tensors have upper indices (receive input)
    - All tensors have lower indices (produce output)
    - Input positions determine which tensors receive input
    """
    
    _EXTRA_PROPS = (
        "_site_tag_id", "_upper_ind_id", "_lower_ind_id", 
        "_L", "_input_positions", "cyclic", "debug", "_phys_dim"
    )
    
    def __init__(self, 
                 arrays, 
                 input_positions=None,
                 shape="lrud", 
                 site_tag_id="I{}", 
                 tags=None,
                 upper_ind_id="k{}", 
                 lower_ind_id="b{}", 
                 bond_name="bond{}",
                 phys_dim=(2, 2),
                 cyclic=False,
                 debug=False,
                 **tn_opts):
        """
        Create an ExpansionMPO.
        
        Args:
            arrays: Tensor arrays defining the operator
            input_positions: Which tensors receive input (if None, infer from shapes)
            shape: Tensor shape convention 
            site_tag_id: Format for site tags
            tags: Global tags
            upper_ind_id: Format for upper indices (inputs)
            lower_ind_id: Format for lower indices (outputs)
            bond_name: Format for bond names
            phys_dim: Physical dimensions (up, down)
            cyclic: Whether to use cyclic boundary conditions
            debug: Enable debug output
            **tn_opts: Additional TensorNetwork options
        """
        Model.__init__(self)
        
        if isinstance(arrays, ExpansionMPO):
            qtn.TensorNetwork.__init__(self, arrays)
            return
        
        self.debug = debug
        arrays = tuple(arrays)
        self._L = len(arrays)
        self._phys_dim = phys_dim
        
        # Process indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        
        # Determine which tensors have upper indices (inputs)
        if input_positions is None:
            # Auto-detect based on array shapes
            self._input_positions = []
            for i, arr in enumerate(arrays):
                # Check if tensor has 4 dimensions (includes upper index)
                expected_dims = 4 if (i > 0 and i < self._L - 1) else 3
                if arr.ndim == expected_dims or (i == 0 and arr.ndim == 2):
                    # Edge case for first/last tensor
                    if i == 0 or i == self._L - 1:
                        # For edge tensors, check if last dimension matches phys_dim[0]
                        if arr.shape[-2] == phys_dim[0] and arr.shape[-1] == phys_dim[1]:
                            self._input_positions.append(i)
                    else:
                        # Middle tensor with 4 dims
                        self._input_positions.append(i)
        else:
            self._input_positions = list(input_positions)
        
        # Store cyclic flag
        self.cyclic = cyclic
        
        if self.debug:
            print(f"[ExpansionMPO] Creating with L={self._L}")
            print(f"[ExpansionMPO] Input positions: {self._input_positions}")
            print(f"[ExpansionMPO] Cyclic: {self.cyclic}")
        
        # Process tags
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
        
        # Build tensors
        tensors = []
        
        for i in range(self._L):
            # Get array
            array = arrays[i]
            
            # Determine if this tensor has input
            has_input = i in self._input_positions
            
            # Build indices for this position
            inds = []
            
            # Left bond
            if i == 0:
                if self.cyclic:
                    inds.append(f"bond{self._L}")
            else:
                inds.append(f"bond{i-1}")
            
            # Right bond
            if i == self._L - 1:
                if self.cyclic:
                    inds.append(f"bond{self._L}")
            else:
                inds.append(f"bond{i}")
            
            # Physical indices
            if has_input:
                inds.append(upper_ind_id.format(i))  # Upper (input)
            inds.append(lower_ind_id.format(i))      # Lower (output)
            
            # Handle edge cases for array shape
            if i == 0 and not self.cyclic:
                # First tensor in open boundary
                if has_input:
                    # Shape should be (right, up, down)
                    if array.ndim == 4:
                        array = array[0]  # Remove trivial left bond
                    assert array.ndim == 3, f"First tensor with input should be 3D, got {array.ndim}D"
                else:
                    # Shape should be (right, down)
                    if array.ndim == 3:
                        array = array[0]  # Remove trivial left bond
                    assert array.ndim == 2, f"First tensor without input should be 2D, got {array.ndim}D"
            
            elif i == self._L - 1 and not self.cyclic:
                # Last tensor in open boundary
                if has_input:
                    # Shape should be (left, up, down)
                    if array.ndim == 4:
                        array = array[:, 0]  # Remove trivial right bond
                    assert array.ndim == 3, f"Last tensor with input should be 3D, got {array.ndim}D"
                else:
                    # Shape should be (left, down)
                    if array.ndim == 3:
                        array = array[:, 0]  # Remove trivial right bond
                    assert array.ndim == 2, f"Last tensor without input should be 2D, got {array.ndim}D"
            
            # Create site tags
            site_tag = site_tag_id.format(i)
            if tags is not None:
                site_tags = (site_tag,) + tags
            else:
                site_tags = site_tag
            
            # Create tensor - no need for transpose, just use natural order
            tensor = qtn.Tensor(data=array, inds=inds, tags=site_tags)
            tensors.append(tensor)
            
            if self.debug and i < 3:  # Show first few tensors
                print(f"[ExpansionMPO] Tensor {i}: shape={array.shape}, "
                      f"inds={inds}, has_input={has_input}")
        
        # Initialize TensorNetwork (without passing cyclic - it's stored as attribute)
        qtn.TensorNetwork.__init__(self, tensors, virtual=True)
    
    @property
    def spacing(self) -> Optional[int]:
        """Spacing between input positions (if regular)."""
        if len(self._input_positions) < 2:
            return None
        
        spacings = [self._input_positions[i+1] - self._input_positions[i] 
                   for i in range(len(self._input_positions)-1)]
        
        # Check if all spacings are equal
        if all(s == spacings[0] for s in spacings):
            return spacings[0]
        return None
    
    @property
    def input_positions(self) -> List[int]:
        """Positions of tensors that receive input."""
        return self._input_positions
    
    @property
    def upper_inds(self):
        """Iterator over upper indices (only at input positions)."""
        for i in self._input_positions:
            yield self.upper_ind(i)
    
    @property
    def lower_inds(self):
        """Iterator over all lower indices."""
        for i in range(self.L):
            yield self.lower_ind(i)
    
    @property
    def phys_dim(self):
        """Physical dimensions (upper, lower)."""
        return self._phys_dim
    
    def upper_ind(self, site):
        """Get upper index name for site."""
        return self._upper_ind_id.format(site)
    
    def lower_ind(self, site):
        """Get lower index name for site."""
        return self._lower_ind_id.format(site)
    
    def site_tag(self, site):
        """Get site tag for site."""
        return self._site_tag_id.format(site)
    
    @debug_timer
    @debug_trace
    def apply(self, other, compress=False, **compress_opts):
        """
        Apply this ExpansionMPO to an MPS or another MPO.
        
        For expansion: few inputs -> many outputs
        The input MPS has fewer sites than this MPO has outputs.
        """
        if isinstance(other, MatrixProductState):
            return self.apply_mps(other, compress=compress, **compress_opts)
        else:
            raise NotImplementedError("ExpansionMPO can only be applied to MPS currently")
    
    def apply_mps(self, mps, compress=False, **compress_opts):
        """
        Apply expansion to an MPS.
        
        The input MPS has N sites, we have M > N output sites.
        Each input connects to specific positions in the expansion.
        """
        if self.debug:
            print(f"\n[ExpansionMPO.apply_mps] Input MPS: L={mps.L}")
            print(f"[ExpansionMPO.apply_mps] Expansion: L={self.L}, "
                  f"inputs at positions {self._input_positions}")
        
        # Check dimensions
        if mps.L != len(self._input_positions):
            raise ValueError(f"Input MPS has {mps.L} sites but "
                           f"ExpansionMPO expects {len(self._input_positions)} inputs")
        
        # Step 1: Contract each input with its corresponding expansion position
        result_tensors = []
        
        for i in range(self.L):
            if i in self._input_positions:
                # This position gets input
                mps_idx = self._input_positions.index(i)
                
                # Get copies
                exp_tensor = self[i].copy()
                mps_tensor = mps[mps_idx].copy()
                
                # Reindex MPS physical index to match
                # Find MPS physical index (the one with size 2)
                mps_phys_ind = None
                for ind in mps_tensor.inds:
                    if mps_tensor.ind_size(ind) == self._phys_dim[0]:
                        # Check it's not a bond
                        if ind.startswith('k'):
                            mps_phys_ind = ind
                            break
                
                if mps_phys_ind is None:
                    # Fallback: smallest dimension
                    sizes = [(ind, mps_tensor.ind_size(ind)) for ind in mps_tensor.inds]
                    sizes.sort(key=lambda x: x[1])
                    mps_phys_ind = sizes[0][0]
                
                # Reindex to match expansion
                exp_phys_ind = f'k{i}'
                mps_tensor.reindex_({mps_phys_ind: exp_phys_ind})
                
                # Contract these two tensors
                tn = qtn.TensorNetwork([exp_tensor, mps_tensor])
                tn.contract_ind(exp_phys_ind)
                
                # Result should be a single tensor
                result = tn.tensors[0]
                result_tensors.append(result)
            else:
                # This position has no input - use original expansion tensor
                result_tensors.append(self[i].copy())
        
        # Step 2: Build tensor network and contract MPS bonds
        result_tn = qtn.TensorNetwork(result_tensors)
        
        # Find MPS bonds - they appear in exactly 2 tensors and aren't our standard indices
        mps_bonds = []
        for ind in result_tn.ind_map:
            if not any(ind.startswith(prefix) for prefix in ['v', 'k', 'b', 'bond']):
                count = sum(1 for t in result_tn if ind in t.inds)
                if count == 2:
                    mps_bonds.append(ind)
        
        # Contract MPS bonds
        for bond in mps_bonds:
            result_tn.contract_ind(bond)
        
        # Step 3: Extract arrays and build output MPS
        output_arrays = []
        
        # We should now have exactly L tensors, one per output position
        if len(result_tn.tensors) != self.L:
            raise ValueError(f"Expected {self.L} tensors after contraction, got {len(result_tn.tensors)}")
        
        # Sort tensors by their lower indices
        for i in range(self.L):
            lower_ind = f'b{i}'
            
            # Find tensor with this lower index
            found_tensor = None
            for t in result_tn.tensors:
                if lower_ind in t.inds:
                    found_tensor = t
                    break
            
            if found_tensor is None:
                raise ValueError(f"Could not find tensor for position {i}")
            
            # Extract array and ensure correct shape
            array = found_tensor.data
            
            # Standard MPS tensor should have 3 dims (or 2 for edges)
            # Current tensor might have bond indices in different order
            if array.ndim == 3:
                # Find physical dimension (should be size 2)
                phys_axis = None
                for ax in range(3):
                    if array.shape[ax] == self._phys_dim[1]:
                        phys_axis = ax
                        break
                
                # Ensure physical is last
                if phys_axis is not None and phys_axis != 2:
                    if phys_axis == 0:
                        array = array.transpose(1, 2, 0)
                    elif phys_axis == 1:
                        array = array.transpose(0, 2, 1)
            elif array.ndim == 2:
                # Edge tensor - add virtual dimension
                if i == 0 and not self.cyclic:
                    array = jnp.expand_dims(array, axis=0)
                elif i == self.L - 1 and not self.cyclic:
                    array = jnp.expand_dims(array, axis=1)
            else:
                raise ValueError(f"Unexpected array shape at position {i}: {array.shape}")
            
            output_arrays.append(array)
        
        # Create output MPS
        from quimb.tensor.tensor_1d import MatrixProductState
        output_mps = MatrixProductState(output_arrays, shape='lrp')
        
        if self.debug:
            print(f"[ExpansionMPO.apply_mps] Output MPS: L={output_mps.L}")
            print(f"[ExpansionMPO.apply_mps] Output norm: {output_mps.norm():.6f}")
        
        # Optionally compress
        if compress:
            output_mps.compress(**compress_opts)
        
        return output_mps
    
    def normalize(self, insert=None):
        """Normalize the expansion MPO."""
        norm = self.norm()
        
        if insert is None:
            # Distribute normalization
            for tensor in self.tensors:
                tensor.modify(data=tensor.data / (norm ** (1/self.L)))
        else:
            # Normalize at specific position
            self.tensors[insert].modify(data=self.tensors[insert].data / norm)
    
    def norm(self, **contract_opts):
        """Calculate norm of the expansion MPO."""
        conj = self.conj()
        # Contract with conjugate
        tn = conj | self
        
        # Contract matching indices
        for i in range(self.L):
            # Contract upper indices if present
            if i in self._input_positions:
                upper_ind = self.upper_ind(i)
                if upper_ind in tn.ind_map:
                    # Get matching conjugate index
                    conj_ind = upper_ind + "*"
                    if conj_ind in tn.ind_map:
                        tn.contract_ind({upper_ind, conj_ind})
            
            # Contract lower indices
            lower_ind = self.lower_ind(i)
            if lower_ind in tn.ind_map:
                conj_ind = lower_ind + "*"
                if conj_ind in tn.ind_map:
                    tn.contract_ind({lower_ind, conj_ind})
        
        # Contract remaining bonds
        result = tn.contract(**contract_opts)
        return jnp.sqrt(jnp.abs(result))
    
    def copy(self):
        """Create a copy of this ExpansionMPO."""
        # Copy tensor data
        new_arrays = []
        for t in self.tensors:
            new_arrays.append(t.data.copy())
        
        # Create new ExpansionMPO with same structure
        return ExpansionMPO(
            arrays=new_arrays,
            input_positions=self._input_positions.copy(),
            shape="lrud",
            site_tag_id=self._site_tag_id,
            upper_ind_id=self._upper_ind_id,
            lower_ind_id=self._lower_ind_id,
            phys_dim=self._phys_dim,
            cyclic=self.cyclic,
            debug=self.debug
        )
    
    def get_debug_info(self) -> dict:
        """Get debug information about this expansion MPO."""
        return {
            'L': self.L,
            'input_positions': self._input_positions,
            'spacing': self.spacing,
            'cyclic': self.cyclic,
            'num_inputs': len(self._input_positions),
            'num_outputs': self.L,
            'expansion_ratio': self.L / len(self._input_positions) if self._input_positions else 0
        }


def expansion_mpo_initialize(
    L: int,
    num_inputs: int,
    initializer,
    key: Any,
    spacing: Optional[int] = None,
    input_positions: Optional[List[int]] = None,
    bond_dim: int = 4,
    phys_dim: Tuple[int, int] = (2, 2),
    cyclic: bool = False,
    debug: bool = False,
    **kwargs
) -> ExpansionMPO:
    """
    Initialize an ExpansionMPO with specified parameters.
    
    Args:
        L: Total number of output tensors
        num_inputs: Number of input positions
        initializer: JAX initializer
        key: Random key
        spacing: Regular spacing between inputs (if None, calculate from L and num_inputs)
        input_positions: Explicit input positions (overrides spacing)
        bond_dim: Virtual bond dimension
        phys_dim: Physical dimensions (up, down)
        cyclic: Whether to use cyclic boundaries
        debug: Enable debug output
        **kwargs: Additional arguments for ExpansionMPO
    
    Returns:
        Initialized ExpansionMPO
    """
    if debug:
        print(f"\n[expansion_mpo_initialize] Creating ExpansionMPO:")
        print(f"  L={L}, num_inputs={num_inputs}")
        print(f"  bond_dim={bond_dim}, phys_dim={phys_dim}")
        print(f"  cyclic={cyclic}")
    
    # Consistency check
    if num_inputs > L:
        raise ValueError(f"num_inputs ({num_inputs}) cannot be greater than L ({L}). "
                        f"Expansion requires fewer inputs than outputs.")
    
    if num_inputs <= 0:
        raise ValueError("num_inputs must be > 0")
    
    # Determine input positions
    if input_positions is None:
        if spacing is None:
            # Calculate even spacing
            if num_inputs > 0:
                spacing = max(1, L // num_inputs)
            else:
                raise ValueError("num_inputs must be > 0")
        
        # Generate positions with given spacing
        input_positions = []
        pos = 0
        while len(input_positions) < num_inputs and pos < L:
            input_positions.append(pos)
            pos += spacing
        
        # Adjust if we have too few
        if len(input_positions) < num_inputs:
            # Distribute remaining positions
            remaining = num_inputs - len(input_positions)
            step = max(1, (L - input_positions[-1] - 1) // (remaining + 1))
            pos = input_positions[-1] + step
            while len(input_positions) < num_inputs and pos < L:
                input_positions.append(pos)
                pos += step
    
    if debug:
        print(f"  Input positions: {input_positions}")
    
    # Create tensor arrays
    arrays = []
    keys = jax.random.split(key, L)
    
    for i in range(L):
        # Determine shape based on position and whether it has input
        has_input = i in input_positions
        
        # Base shape determination
        if i == 0:
            if cyclic:
                # First tensor in cyclic: full bonds
                if has_input:
                    shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (bond_dim, bond_dim, phys_dim[1])
            else:
                # First tensor in open: no left bond
                if has_input:
                    shape = (bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (bond_dim, phys_dim[1])
        elif i == L - 1:
            if cyclic:
                # Last tensor in cyclic: full bonds
                if has_input:
                    shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (bond_dim, bond_dim, phys_dim[1])
            else:
                # Last tensor in open: no right bond
                if has_input:
                    shape = (bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (bond_dim, phys_dim[1])
        else:
            # Middle tensors: full bonds
            if has_input:
                shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
            else:
                shape = (bond_dim, bond_dim, phys_dim[1])
        
        # Initialize tensor
        array = initializer(keys[i], shape)
        
        # For open boundaries, add trivial dimensions to maintain consistency
        if not cyclic:
            if i == 0 and len(shape) < 4:
                # Add trivial left dimension
                array = jnp.expand_dims(array, axis=0)
            elif i == L - 1 and len(shape) < 4:
                # Add trivial right dimension  
                array = jnp.expand_dims(array, axis=1)
        
        arrays.append(array)
        
        if debug and i < 3:
            print(f"  Tensor {i}: shape={shape}, has_input={has_input}")
    
    # Create ExpansionMPO
    expansion = ExpansionMPO(
        arrays,
        input_positions=input_positions,
        phys_dim=phys_dim,
        cyclic=cyclic,
        debug=debug,
        **kwargs
    )
    
    # Normalize
    expansion.normalize()
    
    if debug:
        print(f"[expansion_mpo_initialize] Created ExpansionMPO with norm={expansion.norm():.6f}")
    
    return expansion