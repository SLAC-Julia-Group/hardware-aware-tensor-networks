"""
ExpansionMPO - Fixed implementation with proper contraction logic.
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
    
    This is fundamentally different from SMPO:
    - SMPO: All tensors have inputs, some have outputs
    - ExpansionMPO: Some tensors have inputs, all have outputs
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
        
        if self.debug:
            print(f"\n[ExpansionMPO.__init__] Creating with L={self._L}")
            print(f"[ExpansionMPO.__init__] Input positions: {self._input_positions}")
            print(f"[ExpansionMPO.__init__] Cyclic: {self.cyclic}")
        
        # Process tags
        site_tags = [site_tag_id.format(i) for i in range(self.L)]
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = [(st,) + tags for st in site_tags]

        # Define all order tuples for clarity
        d_ord = (0,)                                          # 1D: (down)
        ld_ord = tuple(shape.replace("r", "").replace("u", "").find(x) for x in "ld")      # 2D: (left, down)
        rd_ord = tuple(shape.replace("l", "").replace("u", "").find(x) for x in "rd")      # 2D: (right, down)  
        lud_ord = tuple(shape.replace("r", "").find(x) for x in "lud")                     # 3D: (left, up, down)
        rud_ord = tuple(shape.replace("l", "").find(x) for x in "rud")                     # 3D: (right, up, down)
        lrd_ord = tuple(shape.replace("u", "").find(x) for x in "lrd")                     # 3D: (left, right, down)
        lrud_ord = tuple(map(shape.find, "lrud"))                                          # 4D: (left, right, up, down)

        # Determine orders based on position, boundary conditions, and input presence
        orders = []
        for i in range(self.L):
            has_input = i in input_positions
            
            if self.cyclic:
                # Cyclic: all tensors have both bonds
                orders.append(lrud_ord if has_input else lrd_ord)
            else:
                # Open boundaries: edge tensors have one bond
                if i == 0:  # First tensor
                    if has_input:
                        orders.append(rud_ord)  # (right, up, down)
                    else:
                        orders.append(rd_ord)   # (right, down)
                elif i == self.L - 1:  # Last tensor
                    if has_input:
                        orders.append(lud_ord)  # (left, up, down)
                    else:
                        orders.append(ld_ord)   # (left, down)
                else:  # Middle tensors
                    if has_input:
                        orders.append(lrud_ord)  # (left, right, up, down)
                    else:
                        orders.append(lrd_ord)   # (left, right, down)
        
        self._orders = orders

        # Build indices for each tensor
        inds = []
        for i in range(self.L):
            tensor_inds = []
            
            # Bonds
            if self.cyclic:
                # Cyclic: all tensors have both bonds
                if i == 0:
                    tensor_inds.extend([f"bond{self.L}", f"bond{i}"])
                elif i == self.L - 1:
                    tensor_inds.extend([f"bond{i-1}", f"bond{self.L}"])
                else:
                    tensor_inds.extend([f"bond{i-1}", f"bond{i}"])
            else:
                # Open: edge tensors have one bond
                if i == 0:
                    tensor_inds.append(f"bond{i}")  # Only right bond
                elif i == self.L - 1:
                    tensor_inds.append(f"bond{i-1}")  # Only left bond
                else:
                    tensor_inds.extend([f"bond{i-1}", f"bond{i}"])  # Both bonds
            
            # Physical indices
            if i in input_positions:
                tensor_inds.append(upper_ind_id.format(i))  # Upper (input)
            tensor_inds.append(lower_ind_id.format(i))      # Lower (output) - always present
            
            inds.append(tuple(tensor_inds))
            
            if self.debug and i < 3:
                print(f"[ExpansionMPO.__init__] Tensor {i}: inds={tensor_inds}, order={orders[i]}, shape={arrays[i].shape}")
        
        # Create tensors with proper transposition
        tensors = []
        for i, (array, site_tag, ind, order) in enumerate(zip(arrays, site_tags, inds, orders)):
            # Validate array shape matches expected indices
            expected_ndim = len(ind)
            if array.ndim != expected_ndim:
                raise ValueError(f"Tensor {i} has {array.ndim} dimensions but expected {expected_ndim} based on indices {ind}")
            
            # Apply transposition based on order
            transposed_array = jnp.transpose(array, order)
            tensor = qtn.Tensor(data=transposed_array, inds=ind, tags=site_tag)
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
    def apply_mps(self, mps, compress=False, normalize_on_contract=True, **compress_opts):
        """
        Apply expansion to MPS through straightforward tensor contractions.
        
        The operation maps N input sites to L output sites by contracting
        MPS physical indices with ExpansionMPO upper indices at specified positions.
        """
        # Validate dimensions
        if mps.L != len(self._input_positions):
            raise ValueError(f"Input MPS has {mps.L} sites but ExpansionMPO expects {len(self._input_positions)} inputs")
        
        if self.debug:
            print(f"\n[apply_mps] Starting expansion: {mps.L} → {self.L}")
            print(f"[apply_mps] Input positions: {self._input_positions}")
        
        # Create a working copy of the ExpansionMPO tensor network
        expansion_copy = self.copy()
        mps_copy = mps.copy()
        
        # Global index renaming to avoid any conflicts
        # Rename all ExpansionMPO indices to have 'exp_' prefix
        for i, tensor in enumerate(expansion_copy.tensors):
            new_inds = {}
            for ind in tensor.inds:
                if ind.startswith('bond'):
                    new_inds[ind] = f'exp_{ind}'
                elif ind.startswith('b'):
                    new_inds[ind] = f'exp_b{i}'  # Ensure unique lower indices
                elif ind.startswith('k'):
                    new_inds[ind] = f'exp_k{i}'  # Ensure unique upper indices
            tensor.reindex_(new_inds)
        
        # Rename all MPS indices to have 'mps_' prefix
        for i, tensor in enumerate(mps_copy.tensors):
            new_inds = {}
            for ind in tensor.inds:
                if ind.startswith('k'):
                    new_inds[ind] = f'mps_phys_{i}'
                else:
                    new_inds[ind] = f'mps_{i}_{ind}'
            tensor.reindex_(new_inds)
        
        if self.debug:
            print(f"\n[apply_mps] After index renaming:")
            print(f"  ExpansionMPO indices: {[t.inds for t in expansion_copy.tensors[:3]]}...")
            print(f"  MPS indices: {[t.inds for t in mps_copy.tensors]}")
        
        # Build the combined tensor network
        combined_tn = qtn.TensorNetwork(expansion_copy.tensors + mps_copy.tensors)
        
        # Contract MPS physical indices with expansion upper indices
        for mps_idx, exp_pos in enumerate(self._input_positions):
            mps_phys = f'mps_phys_{mps_idx}'
            exp_upper = f'exp_k{exp_pos}'
            
            if self.debug:
                print(f"\n[apply_mps] Contracting MPS[{mps_idx}] with Expansion[{exp_pos}]")
                print(f"  {mps_phys} ← → {exp_upper}")
            
            # Reindex to prepare for contraction
            if mps_phys in combined_tn.ind_map and exp_upper in combined_tn.ind_map:
                combined_tn.reindex_({mps_phys: exp_upper})
                combined_tn.contract_ind(exp_upper)
            else:
                raise ValueError(f"Missing indices for contraction: {mps_phys} or {exp_upper}")
        
        # Contract any remaining MPS bonds (these are internal to the MPS)
        mps_internal_bonds = set()
        for ind in combined_tn.ind_map:
            if ind.startswith('mps_') and not ind.startswith('mps_phys_'):
                if len(combined_tn.ind_map[ind]) == 2:  # Appears in exactly 2 tensors
                    mps_internal_bonds.add(ind)
        
        if self.debug and mps_internal_bonds:
            print(f"\n[apply_mps] Contracting MPS internal bonds: {mps_internal_bonds}")
        
        for bond in mps_internal_bonds:
            combined_tn.contract_ind(bond)
        
        # Extract the output tensors in order
        output_tensors = []
        for i in range(self.L):
            # Find tensor with lower index exp_b{i}
            lower_ind = f'exp_b{i}'
            found = False
            
            for tensor in combined_tn:
                if lower_ind in tensor.inds:
                    output_tensors.append(tensor)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Could not find output tensor for position {i} with index {lower_ind}")
        
        if self.debug:
            print(f"\n[apply_mps] Extracted {len(output_tensors)} output tensors")
        
        # Build output MPS arrays
        output_arrays = []
        
        for i, tensor in enumerate(output_tensors):
            array = tensor.data
            
            # Identify the physical index (the lower index)
            lower_ind = f'exp_b{i}'
            if lower_ind not in tensor.inds:
                raise ValueError(f"Output tensor {i} missing physical index {lower_ind}")
            
            phys_pos = list(tensor.inds).index(lower_ind)
            
            # Reorder dimensions for MPS format based on position
            if not self.cyclic:
                if i == 0:
                    # First tensor: (bond, physical)
                    if array.ndim == 2 and phys_pos == 0:
                        array = array.T
                    elif array.ndim > 2:
                        array = jnp.squeeze(array)
                        if array.ndim == 2 and phys_pos == 0:
                            array = array.T
                elif i == self.L - 1:
                    # Last tensor: (bond, physical)
                    if array.ndim == 2 and phys_pos == 0:
                        array = array.T
                    elif array.ndim > 2:
                        array = jnp.squeeze(array)
                        if array.ndim == 2 and phys_pos == 0:
                            array = array.T
                else:
                    # Middle tensor: (left_bond, right_bond, physical)
                    if array.ndim == 3 and phys_pos != 2:
                        if phys_pos == 0:
                            array = jnp.transpose(array, (1, 2, 0))
                        elif phys_pos == 1:
                            array = jnp.transpose(array, (0, 2, 1))
            else:
                # Cyclic: all tensors are 3D with physical last
                if array.ndim == 3 and phys_pos != 2:
                    if phys_pos == 0:
                        array = jnp.transpose(array, (1, 2, 0))
                    elif phys_pos == 1:
                        array = jnp.transpose(array, (0, 2, 1))
            
            output_arrays.append(array)
            
            if self.debug:
                print(f"  Output tensor {i}: shape {array.shape}")
        
        # Create output MPS
        from quimb.tensor.tensor_1d import MatrixProductState
        output_mps = MatrixProductState(output_arrays, shape='lrp', cyclic=self.cyclic)
        
        if self.debug:
            print(f"\n[apply_mps] Success! Created MPS with L={output_mps.L}, norm={output_mps.norm():.6f}")
        
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
        if self.L > 200:  # for large systems
            for i, tensor in enumerate(self.tensors):
                if i == 0:
                    self.left_canonize_site(i)
                elif i == self.L - 1:
                    tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                else:
                    tensor.modify(data=tensor.data / jnp.linalg.norm(tensor.data))
                    self.left_canonize_site(i)
        else:
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

    # Properties
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
        # Cyclic: all tensors have both bonds
        if has_input:
            shape = (bond_dim, bond_dim, *phys_dim)  # lrud
        else:
            shape = (bond_dim, bond_dim, phys_dim[1])  # lrd
    else:
        # Open boundaries: edge tensors have fewer dimensions
        if position == 1:  # First tensor (1-indexed)
            if has_input:
                shape = (bond_dim, *phys_dim)  # rud: (right, up, down)
            else:
                shape = (bond_dim, phys_dim[1])  # rd: (right, down)
        elif position == L:  # Last tensor
            if has_input:
                shape = (bond_dim, *phys_dim)  # lud: (left, up, down)
            else:
                shape = (bond_dim, phys_dim[1])  # ld: (left, down)
        else:  # Middle tensor
            if has_input:
                shape = (bond_dim, bond_dim, *phys_dim)  # lrud
            else:
                shape = (bond_dim, bond_dim, phys_dim[1])  # lrd
    
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
        # Distribute inputs evenly
        if num_inputs == 1:
            input_positions = [L // 2]  # Middle position
        else:
            # Even spacing
            step = L / (num_inputs + 1)
            input_positions = [int((i + 1) * step) for i in range(num_inputs)]
            # Ensure within bounds
            input_positions = [min(max(0, p), L-1) for p in input_positions]
    
    # Validate input positions
    if len(input_positions) != num_inputs:
        raise ValueError(f"Length of input_positions ({len(input_positions)}) must equal num_inputs ({num_inputs})")
    
    if any(p < 0 or p >= L for p in input_positions):
        raise ValueError(f"All input positions must be in range [0, {L-1}]")
    
    if len(set(input_positions)) != len(input_positions):
        raise ValueError("Input positions must be unique")
    
    if debug:
        print(f"\n[expansion_mpo_initialize] Creating ExpansionMPO:")
        print(f"  L={L}, num_inputs={num_inputs}")
        print(f"  input_positions={input_positions}")
        print(f"  bond_dim={bond_dim}, phys_dim={phys_dim}")
        print(f"  cyclic={cyclic}, boundary={boundary}")
    
    # Generate tensor arrays
    arrays = []
    keys = jax.random.split(key, L)
    
    for i in range(L):
        has_input = i in input_positions
        
        # Generate shape (using 1-indexed position for compatibility)
        shape = generate_expansion_shape(
            method=shape_method,
            L=L,
            has_input=has_input,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            cyclic=cyclic,
            position=i + 1  # Convert to 1-indexed
        )
        
        # Initialize array
        array = initializer(keys[i], shape, jnp.float32)
        
        # Add identity if requested (for testing)
        if add_identity and has_input and len(shape) >= 3:
            # Add identity to the upper-lower connection
            if len(shape) == 4:  # Middle tensor with input
                identity = jnp.eye(min(shape[2], shape[3]), dtype=array.dtype)
                array = array.at[:, :, :identity.shape[0], :identity.shape[1]].add(identity)
            elif len(shape) == 3 and i == 0:  # First tensor with input
                identity = jnp.eye(min(shape[1], shape[2]), dtype=array.dtype)
                array = array.at[:, :identity.shape[0], :identity.shape[1]].add(identity)
            elif len(shape) == 3 and i == L-1:  # Last tensor with input
                identity = jnp.eye(min(shape[1], shape[2]), dtype=array.dtype)
                array = array.at[:, :identity.shape[0], :identity.shape[1]].add(identity)
        
        arrays.append(array)
        
        if debug and i < 3:
            print(f"  Tensor {i}: has_input={has_input}, shape={shape}")
    
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
    
    if debug:
        print(f"[expansion_mpo_initialize] Created ExpansionMPO with norm={expansion.norm():.6f}")
    
    return expansion