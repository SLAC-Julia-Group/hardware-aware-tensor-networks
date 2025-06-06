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
import autoray as a

from tn4ml.models.model import Model
from tn4ml.util import return_digits
from ..core.base import debug_timer, debug_trace


class ExpansionMPO(TensorNetwork1DOperator, TensorNetwork1DFlat, Model):
    """
    An MPO for expansion operations - the reverse of SMPO.
    
    Instead of having many inputs with few outputs (compression),
    this has few inputs with many outputs (expansion).
    
    Key differences from SMPO:
    - Only some tensors have upper indices (receive input)
    - All tensors have lower indices (produce output)
    - Spacing determines which tensors receive input
    """
    
    _EXTRA_PROPS = (
        "_site_tag_id", "_upper_ind_id", "_lower_ind_id", 
        "_L", "_spacing", "_input_positions", "cyclic", "debug"
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
                 debug=False,
                 **tn_opts):
        """
        Create an ExpansionMPO.
        
        Args:
            arrays: Tensor arrays defining the operator
            input_positions: Which tensors receive input (if None, use even spacing)
            shape: Tensor shape convention 
            site_tag_id: Format for site tags
            tags: Global tags
            upper_ind_id: Format for upper indices (inputs)
            lower_ind_id: Format for lower indices (outputs)
            bond_name: Format for bond names
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
        
        # Process indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        
        # All tensors have lower indices (outputs)
        lower_inds = [lower_ind_id.format(i) for i in range(self.L)]
        
        # Determine which tensors have upper indices (inputs)
        if input_positions is None:
            # Auto-calculate based on array shapes
            self._input_positions = []
            for i, arr in enumerate(arrays):
                if arr.ndim == 4:  # Has upper index
                    self._input_positions.append(i)
        else:
            self._input_positions = input_positions
        
        # Calculate spacing if positions are regular
        if len(self._input_positions) > 1:
            spacings = [self._input_positions[i+1] - self._input_positions[i] 
                       for i in range(len(self._input_positions)-1)]
            if all(s == spacings[0] for s in spacings):
                self._spacing = spacings[0]
            else:
                self._spacing = None  # Irregular spacing
        else:
            self._spacing = self._L  # Single input
        
        if self.debug:
            print(f"[ExpansionMPO] Creating with L={self._L}")
            print(f"[ExpansionMPO] Input positions: {self._input_positions}")
            print(f"[ExpansionMPO] Spacing: {self._spacing}")
        
        # Upper indices only for input positions
        upper_inds = [upper_ind_id.format(i) if i in self._input_positions else None 
                     for i in range(self.L)]
        
        # Process tags
        self._site_tag_id = site_tag_id
        site_tags = [site_tag_id.format(i) for i in range(self.L)]
        if tags is not None:
            tags = (tags,) if isinstance(tags, str) else tuple(tags)
            site_tags = [(st,) + tags for st in site_tags]
        
        # Determine if cyclic
        self.cyclic = arrays[0].ndim == 4 or arrays[-1].ndim == 4
        
        # Process tensor orders based on shape
        lu_ord = tuple(shape.replace("r", "").replace("d", "").find(x) for x in "lu")
        ld_ord = tuple(shape.replace("r", "").replace("u", "").find(x) for x in "ld")
        lud_ord = tuple(shape.replace("r", "").find(x) for x in "lud")
        lrd_ord = tuple(shape.replace("u", "").find(x) for x in "lrd")
        lrud_ord = tuple(map(shape.find, "lrud"))
        
        # Build orders for each tensor
        orders = []
        for i in range(self.L):
            if i == 0:
                if self.cyclic:
                    if i in self._input_positions:
                        orders.append(lrud_ord)
                    else:
                        orders.append(lrd_ord)
                else:
                    if i in self._input_positions:
                        orders.append((1, 2, 0))  # rud order
                    else:
                        orders.append((1, 0))     # rd order
            elif i == self.L - 1:
                if self.cyclic:
                    if i in self._input_positions:
                        orders.append(lrud_ord)
                    else:
                        orders.append(lrd_ord)
                else:
                    if i in self._input_positions:
                        orders.append(lud_ord)
                    else:
                        orders.append(ld_ord)
            else:
                if i in self._input_positions:
                    orders.append(lrud_ord)
                else:
                    orders.append(lrd_ord)
        
        self._orders = orders
        
        # Build tensor indices
        cyc_bond = (f"bond_{self.L}",) if self.cyclic else ()
        
        inds = []
        for i in range(self.L):
            # Determine bonds
            if i == 0:
                left_bond = cyc_bond if self.cyclic else ()
                right_bond = (f"bond_{i}",) if i < self.L - 1 else cyc_bond
            elif i == self.L - 1:
                left_bond = (f"bond_{i-1}",)
                right_bond = cyc_bond if self.cyclic else ()
            else:
                left_bond = (f"bond_{i-1}",)
                right_bond = (f"bond_{i}",)
            
            # Build indices for this tensor
            tensor_inds = left_bond + right_bond
            if upper_inds[i] is not None:
                tensor_inds += (upper_inds[i],)
            tensor_inds += (lower_inds[i],)
            
            inds.append(tensor_inds)
        
        # Create tensors
        tensors = []
        for i, (array, site_tag, ind, order) in enumerate(zip(arrays, site_tags, inds, orders)):
            # Transpose array according to order
            transposed = a.do("transpose", array, order)
            tensor = qtn.Tensor(data=transposed, inds=ind, tags=site_tag)
            tensors.append(tensor)
            
            if self.debug and i < 3:  # Show first few tensors
                print(f"[ExpansionMPO] Tensor {i}: shape={array.shape}, "
                      f"inds={ind}, has_input={i in self._input_positions}")
        
        qtn.TensorNetwork.__init__(self, tensors, virtual=True, **tn_opts)
    
    @property
    def spacing(self) -> Optional[int]:
        """Spacing between input positions (if regular)."""
        return self._spacing
    
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
        expansion, input_mps = self.copy(), mps.copy()
        
        if self.debug:
            print(f"\n[ExpansionMPO.apply_mps] Input MPS: L={len(input_mps.tensors)}")
            print(f"[ExpansionMPO.apply_mps] Expansion: L={self.L}, "
                  f"inputs at positions {self._input_positions}")
        
        # Check dimensions
        if len(input_mps.tensors) != len(self._input_positions):
            raise ValueError(f"Input MPS has {len(input_mps.tensors)} sites but "
                           f"ExpansionMPO expects {len(self._input_positions)} inputs")
        
        # Align indices: map input MPS sites to expansion positions
        for i, (mps_site, exp_pos) in enumerate(zip(range(len(input_mps.tensors)), 
                                                    self._input_positions)):
            # Get the physical index from input MPS tensor
            input_tensor = input_mps[mps_site]
            
            # Find the physical index (usually the smallest or the one starting with 'k')
            phys_ind = None
            for ind in input_tensor.inds:
                if not any(bond in ind for bond in ['bond', 'v']):
                    phys_ind = ind
                    break
            
            if phys_ind is None:
                # Fallback: smallest dimension
                phys_ind = min(input_tensor.inds, key=lambda x: input_tensor.ind_size(x))
            
            # Rename to match expansion upper index
            new_ind = self.upper_ind(exp_pos)
            input_tensor.reindex_({phys_ind: new_ind})
            
            if self.debug:
                print(f"[ExpansionMPO.apply_mps] Mapped input {mps_site} "
                      f"(ind {phys_ind}) to position {exp_pos} (ind {new_ind})")
        
        # Contract expansion with input
        result = expansion & input_mps
        
        # Contract all upper indices
        for ind in self.upper_inds:
            if ind in result.ind_map:
                result.contract_ind(ind)
        
        # The result should have tensors at all L positions
        # Build output MPS
        output_arrays = []
        
        for i in range(self.L):
            # Get tensor at position i
            tensor = result[self.site_tag(i)]
            
            # Extract and reshape data
            data = tensor.data
            
            # Ensure proper MPS format
            if len(data.shape) > 3:
                # Remove extra dimensions
                data = jnp.squeeze(data)
            
            # Handle edge cases
            if len(data.shape) == 2:
                if i == 0 or i == self.L - 1:
                    # Add virtual dimension
                    data = jnp.expand_dims(data, axis=0 if i == 0 else 1)
                else:
                    # Middle tensor missing dimension - shouldn't happen
                    raise ValueError(f"Tensor {i} has unexpected shape {data.shape}")
            
            output_arrays.append(data)
        
        # Create output MPS
        output_mps = MatrixProductState(output_arrays, shape='lrp')
        
        if self.debug:
            print(f"[ExpansionMPO.apply_mps] Output MPS: L={len(output_mps.tensors)}")
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
        overlap = conj & self
        return jnp.sqrt(overlap.contract(**contract_opts))
    
    def get_debug_info(self) -> dict:
        """Get debug information about this expansion MPO."""
        return {
            'L': self.L,
            'input_positions': self._input_positions,
            'spacing': self._spacing,
            'cyclic': self.cyclic,
            'num_inputs': len(self._input_positions),
            'num_outputs': self.L,
            'expansion_ratio': self.L / len(self._input_positions)
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
    
    # Determine input positions
    if input_positions is None:
        if spacing is None:
            # Calculate even spacing
            spacing = L // num_inputs
            if spacing * num_inputs < L:
                print(f"[WARNING] With L={L} and num_inputs={num_inputs}, "
                      f"spacing={spacing} doesn't divide evenly")
        
        # Generate positions with given spacing
        input_positions = []
        pos = 0
        while len(input_positions) < num_inputs and pos < L:
            input_positions.append(pos)
            pos += spacing
        
        # Adjust if we have too few
        if len(input_positions) < num_inputs:
            print(f"[WARNING] Could only place {len(input_positions)} inputs, "
                  f"adjusting spacing")
            input_positions = list(range(0, L, max(1, L // num_inputs)))[:num_inputs]
    
    if debug:
        print(f"  Input positions: {input_positions}")
    
    # Create tensor arrays
    arrays = []
    keys = jax.random.split(key, L)
    
    for i in range(L):
        # Determine shape based on position
        if i in input_positions:
            # Tensor with input
            if i == 0:
                if cyclic:
                    shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (1, bond_dim, phys_dim[0], phys_dim[1])
            elif i == L - 1:
                if cyclic:
                    shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
                else:
                    shape = (bond_dim, 1, phys_dim[0], phys_dim[1])
            else:
                shape = (bond_dim, bond_dim, phys_dim[0], phys_dim[1])
        else:
            # Tensor without input (only bonds and output)
            if i == 0:
                if cyclic:
                    shape = (bond_dim, bond_dim, phys_dim[1])
                else:
                    shape = (1, bond_dim, phys_dim[1])
            elif i == L - 1:
                if cyclic:
                    shape = (bond_dim, bond_dim, phys_dim[1])
                else:
                    shape = (bond_dim, 1, phys_dim[1])
            else:
                shape = (bond_dim, bond_dim, phys_dim[1])
        
        # Initialize tensor
        array = initializer(keys[i], shape)
        arrays.append(array)
        
        if debug and i < 3:
            print(f"  Tensor {i}: shape={shape}, has_input={i in input_positions}")
    
    # Create ExpansionMPO
    expansion = ExpansionMPO(
        arrays,
        input_positions=input_positions,
        debug=debug,
        **kwargs
    )
    
    # Normalize
    expansion.normalize()
    
    if debug:
        print(f"[expansion_mpo_initialize] Created ExpansionMPO with norm={expansion.norm():.6f}")
    
    return expansion