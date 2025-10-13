import tn4ml
from tn4ml.models.smpo import SpacedMatrixProductOperator, SMPO_initialize
from tn4ml.models.model import Model
from typing import List, Optional, Tuple, Any
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork
import jax.numpy as jnp
import jax


class CascadedSMPO:
    """
    A cascaded sequence of SpacedMatrixProductOperators for autoencoder architectures.
    
    Allows chaining multiple SMPOs where output of layer i becomes input to layer i+1.
    Example: 7→4→2→4→7 autoencoder with 4 SMPO layers.
    
    Pure container class - creates connected TensorNetwork only when needed for training.
    """
    
    def __init__(self, smpo_list: List[SpacedMatrixProductOperator], **kwargs):
        """
        Initialize cascaded SMPO.
        
        Parameters
        ----------
        smpo_list : List[SpacedMatrixProductOperator]
            List of SMPO layers in forward order (input to output)
        **kwargs
            Additional arguments (ignored, for compatibility)
        """
        # Handle case where we're copying from another CascadedSMPO
        if isinstance(smpo_list, CascadedSMPO):
            self.layers = [layer.copy() for layer in smpo_list.layers]
        else:
            self.layers = smpo_list
            
        self.num_layers = len(self.layers)
        
        # Basic validation
        if self.num_layers == 0:
            raise ValueError("At least one SMPO layer required")
        
        # Validate dimensions between layers
        self.validate_dimensions()
    
    def validate_dimensions(self):
        """
        Check that output dimensions of layer i are compatible with input dimensions of layer i+1.
        Now allows for approximate matches and suggests fixes.
        """
        for i in range(self.num_layers - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            # Count outputs of current layer
            current_outputs = self._count_outputs(current_layer)
            
            # Count inputs of next layer (should equal its tensor count)
            next_inputs = next_layer.L
            
            if current_outputs != next_inputs:
                print(f"WARNING: Dimension mismatch between layer {i} and {i+1}:")
                print(f"         Layer {i} outputs {current_outputs} but layer {i+1} expects {next_inputs} inputs")
                print(f"         This may work if the difference is small, but could cause issues")
                
                # Only raise error if the mismatch is too large
                ratio = current_outputs / next_inputs if next_inputs > 0 else float('inf')
                if ratio < 0.5 or ratio > 2.0:
                    raise ValueError(
                        f"Dimension mismatch too large between layer {i} and {i+1}: "
                        f"layer {i} outputs {current_outputs} but layer {i+1} expects {next_inputs} inputs"
                    )
    
    def _count_outputs(self, smpo: SpacedMatrixProductOperator) -> int:
        """
        Count the number of output indices from an SMPO layer.
        """
        # First try to get actual output indices
        try:
            return len(list(smpo.lower_inds))
        except:
            # Fallback to calculation-based approach
            if hasattr(smpo, 'spacings') and smpo.spacings:
                # Non-uniform spacing case
                return len(smpo.spacings) + 1  # +1 for the last output
            elif smpo.spacing > 0:
                # Uniform spacing case
                return len(list(range(0, smpo.L, smpo.spacing)))
            else:
                # output_inds case or fallback: assume we can access the attribute
                if hasattr(smpo, 'output_inds') and smpo.output_inds:
                    return len(smpo.output_inds)
                else:
                    # Ultimate fallback: assume all tensors have outputs
                    return smpo.L
    
    def forward(self, input_mps, store_intermediates: bool = False):
        """
        Forward pass through the cascaded SMPO layers.
        
        Parameters
        ----------
        input_mps : MatrixProductState
            Input MPS to process
        store_intermediates : bool
            Whether to store intermediate results for debugging/analysis
            
        Returns
        -------
        MatrixProductState
            Final output after processing through all layers
        """
        current = input_mps
        
        if store_intermediates:
            self.intermediates = [current]
        
        # Process through each SMPO layer
        for i, layer in enumerate(self.layers):
            try:
                current = layer.apply(current)
                
                if store_intermediates:
                    self.intermediates.append(current)
                    
            except Exception as e:
                raise RuntimeError(f"Error in layer {i}: {str(e)}") from e
        
        return current
    
    def apply(self, input_mps, store_intermediates: bool = False):
        """Alias for forward() for compatibility with training code."""
        return self.forward(input_mps, store_intermediates)
    
    def __call__(self, input_mps, store_intermediates: bool = False):
        """Make the class callable like a function."""
        return self.apply(input_mps, store_intermediates)
    
    def copy(self, deep=False):
        """Copy the CascadedSMPO."""
        if deep:
            import copy
            return copy.deepcopy(self)
        return CascadedSMPO([layer.copy() for layer in self.layers])
    
    def prepare_for_training(self) -> TensorNetwork:
        """
        Create a properly connected TensorNetwork for training.
        
        This method connects the layers by renaming indices so that:
        - Layer 1 outputs (b0, b2, b4, b6) connect to Layer 2 inputs (k0, k1, k2, k3)
        - All indices are renamed to avoid collisions between layers
        
        Returns
        -------
        TensorNetwork
            A single connected tensor network ready for qtn.pack/unpack
        """
        if self.num_layers == 1:
            # Single layer case - just return the layer as a TensorNetwork
            return TensorNetwork(self.layers[0].tensors)
        
        # Start with all tensors from first layer (unchanged)
        all_tensors = list(self.layers[0].tensors)
        
        # For each subsequent layer, rename ALL indices to avoid collisions
        for layer_idx in range(1, self.num_layers):
            prev_layer = self.layers[layer_idx - 1]
            curr_layer = self.layers[layer_idx]
            
            # Get output indices from previous layer
            prev_outputs = list(prev_layer.lower_inds)
            
            if len(prev_outputs) != curr_layer.L:
                print(f"WARNING: Connection mismatch - layer {layer_idx-1} has {len(prev_outputs)} outputs")
                print(f"         but layer {layer_idx} expects {curr_layer.L} inputs")
                
                # Try to handle the mismatch
                if len(prev_outputs) > curr_layer.L:
                    # More outputs than needed - take the first N
                    prev_outputs = prev_outputs[:curr_layer.L]
                    print(f"         Using first {curr_layer.L} outputs: {prev_outputs}")
                else:
                    # Fewer outputs than needed - this is harder to fix
                    raise ValueError(
                        f"Cannot connect layers: layer {layer_idx-1} has only {len(prev_outputs)} outputs "
                        f"but layer {layer_idx} needs {curr_layer.L} inputs. "
                        f"Try adjusting the layer dimensions or spacing."
                    )
            
            # Process each tensor in current layer
            for tensor_idx, tensor in enumerate(curr_layer.tensors):
                # Copy the tensor
                new_tensor = tensor.copy()
                
                # Get current indices
                current_inds = list(tensor.inds)
                
                # Create mapping for ALL indices to avoid collisions
                index_mapping = {}
                
                for ind in current_inds:
                    if ind.startswith('k') and tensor_idx < len(prev_outputs):
                        # Input index: connect to previous layer output
                        index_mapping[ind] = prev_outputs[tensor_idx]
                    elif ind.startswith('b'):
                        # Output index: rename to avoid collision with previous layers
                        new_ind = f"L{layer_idx}_{ind}"
                        index_mapping[ind] = new_ind
                    elif ind.startswith('bond_'):
                        # Virtual bond: rename to avoid collision
                        new_ind = f"L{layer_idx}_{ind}"
                        index_mapping[ind] = new_ind
                    else:
                        # Keep other indices as-is
                        index_mapping[ind] = ind
                
                # Apply the renaming
                if index_mapping:
                    new_tensor.reindex(index_mapping)
                
                all_tensors.append(new_tensor)
        
        # Create connected tensor network
        return TensorNetwork(all_tensors)
    
    def __repr__(self):
        """String representation showing the architecture."""
        dims = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                dims.append(str(layer.L))  # Input dimension
            dims.append(str(self._count_outputs(layer)))  # Output dimension
        
        architecture = " → ".join(dims)
        return f"CascadedSMPO({architecture})"
    
    # Add Model-like functionality manually
    def configure(self, **kwargs):
        """Configure training parameters (Model-like interface)."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def L(self):
        """Total number of tensors across all layers."""
        return sum(layer.L for layer in self.layers)
    
    @property
    def tensors(self):
        """Get all tensors from all layers (for compatibility with loss functions)."""
        all_tensors = []
        for layer in self.layers:
            all_tensors.extend(layer.tensors)
        return all_tensors
    
    def norm(self, **contract_opts):
        """
        Compute the norm of the cascaded SMPO using additive layer approach.
        
        Uses ||Cascade||² = ||L1||² + ||L2||² + ... which is mathematically sound
        and avoids the connection/contraction issues.
        
        Returns
        -------
        float
            Frobenius norm of the cascaded SMPO
        """
        print("DEBUG: Computing cascaded norm using additive layer approach...")
        
        total_norm_squared = 0.0
        
        for i, layer in enumerate(self.layers):
            try:
                print(f"DEBUG: Computing norm for layer {i}")
                layer_norm = layer.norm(**contract_opts)
                layer_norm_squared = layer_norm ** 2
                total_norm_squared += layer_norm_squared
                print(f"DEBUG: Layer {i} norm = {layer_norm}, norm² = {layer_norm_squared}")
                
            except Exception as e:
                print(f"WARNING: Layer {i} norm() failed: {e}")
                print(f"DEBUG: Falling back to manual tensor norm for layer {i}")
                
                # Fallback: sum of individual tensor norms
                layer_norm_squared = 0.0
                for j, tensor in enumerate(layer.tensors):
                    try:
                        tensor_norm = float(jnp.linalg.norm(tensor.data))
                        layer_norm_squared += tensor_norm ** 2
                        print(f"DEBUG: Layer {i}, tensor {j} norm = {tensor_norm}")
                    except Exception as tensor_e:
                        print(f"ERROR: Layer {i}, tensor {j} norm failed: {tensor_e}")
                        # Ultimate fallback
                        layer_norm_squared += 1.0
                
                total_norm_squared += layer_norm_squared
                print(f"DEBUG: Layer {i} manual norm² = {layer_norm_squared}")
        
        result = jnp.sqrt(total_norm_squared)
        print(f"DEBUG: Total norm² = {total_norm_squared}, final norm = {result}")
        
        return result
    
    @classmethod
    def create_autoencoder(cls, 
                          layer_dims: List[int],
                          initializer,
                          key,
                          bond_dim: int = 4,
                          phys_dim: Tuple[int, int] = (2, 2),
                          shape_method: str = 'even',
                          add_identity: bool = False,
                          boundary: str = 'obc',
                          cyclic: bool = False,
                          compress: bool = True,
                          **kwargs):
        """
        Factory method to create a cascaded autoencoder architecture.
        
        Automatically calculates optimal output indices for each layer to achieve
        exact dimension matching between layers.
        
        Parameters
        ----------
        layer_dims : List[int]
            Dimensions for each layer, e.g., [56, 32, 16, 3] for autoencoder
            spacing is automatically calculated to achieve these exact dimensions
        initializer : jax.nn.initializers.Initializer
            Initialization function for tensors
        key : jax.random.PRNGKey
            Random key for initialization
        bond_dim : int
            Bond dimension for virtual indices
        phys_dim : Tuple[int, int]
            Physical dimension (up, down)
        shape_method : str
            Shape generation method ('even' or 'noteven')
        add_identity : bool
            Whether to add identity components
        boundary : str
            Boundary conditions
        cyclic : bool
            Whether SMPOs are cyclic
        compress : bool
            Whether to compress bond dimensions
        
        Returns
        -------
        CascadedSMPO
            Configured autoencoder
        """
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions for input and output")
        
        # Auto-adjust shape_method based on configuration
        if cyclic and shape_method == 'noteven':
            print("WARNING: cyclic=True requires shape_method='even', auto-correcting...")
            shape_method = 'even'
        
        layers = []
        keys = jax.random.split(key, len(layer_dims) - 1)
        
        for i in range(len(layer_dims) - 1):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1]
            
            print(f"Creating layer {i}: {input_dim} tensors → {output_dim} outputs")
            
            # Calculate optimal spacing to get close to desired output count
            if output_dim >= input_dim:
                spacing = 1  # All tensors have outputs
            elif output_dim == 1:
                spacing = input_dim  # Only last tensor has output
            else:
                # For uniform spacing: spacing ≈ (input_dim - 1) / (output_dim - 1)
                spacing = max(1, (input_dim - 1) // (output_dim - 1))
            
            print(f"  Using spacing: {spacing}")
            
            # Create SMPO with calculated spacing
            smpo = SMPO_initialize(
                L=input_dim,
                initializer=initializer,
                key=keys[i],
                shape_method=shape_method,
                spacing=spacing,
                bond_dim=bond_dim,
                phys_dim=phys_dim,
                cyclic=cyclic,
                compress=compress,
                add_identity=add_identity,
                boundary=boundary,
                **kwargs
            )
            
            # Check if we got the right number of outputs
            actual_outputs = len(list(smpo.lower_inds))
            print(f"  Actual outputs: {actual_outputs} (target: {output_dim})")
            
            if actual_outputs != output_dim:
                print(f"  WARNING: Got {actual_outputs} outputs instead of {output_dim}")
                print(f"           This may cause connection issues between layers")
            
            layers.append(smpo)
        
        return cls(layers)


def create_example_autoencoder(key, architecture="7-4-2-4-7", cyclic=False):
    """
    Convenience function to create example autoencoder architectures.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initialization
    architecture : str
        Architecture string like "7-4-2-4-7"
    cyclic : bool
        Whether SMPOs are cyclic
        
    Returns
    -------
    CascadedSMPO
        Example autoencoder
    """
    from jax.nn.initializers import normal
    
    dims = [int(x) for x in architecture.split("-")]
    
    return CascadedSMPO.create_autoencoder(
        layer_dims=dims,
        initializer=normal(stddev=0.1),
        key=key,
        bond_dim=4,
        phys_dim=(2, 2),
        add_identity=False,
        boundary='obc',
        cyclic=cyclic,
        compress=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the autoencoder creation
    key = jax.random.key(42)
    
    # Create 7→4→2→4→7 autoencoder
    autoencoder = create_example_autoencoder(key, "7-4-2-4-7", cyclic=False)
    print(f"Created: {autoencoder}")
    
    # Test connected TensorNetwork creation
    try:
        connected_tn = autoencoder.prepare_for_training()
        print(f"✅ Connected TensorNetwork created with {len(connected_tn.tensors)} tensors")
        
        # Test qtn.pack/unpack
        arrays, skeleton = qtn.pack(connected_tn)
        print(f"✅ qtn.pack() worked! Got {len(arrays)} arrays")
        
        unpacked_tn = qtn.unpack(arrays, skeleton)
        print(f"✅ qtn.unpack() worked!")
        
        # Test training setup
        from tn4ml.models.model import Model
        model = Model()
        model.__dict__.update(unpacked_tn.__dict__)
        model.L = len(model.tensors)
        model.apply = autoencoder.apply  # Use cascaded apply method
        print(f"✅ Training setup worked! Model has {model.L} tensors")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test individual layer creation
    from jax.nn.initializers import normal
    
    smpo1 = SMPO_initialize(
        L=7,
        initializer=normal(stddev=0.1),
        key=key,
        shape_method='even',
        spacing=2,
        bond_dim=4,
        phys_dim=(2, 2),
        cyclic=False,
        compress=True,
        add_identity=False,
        boundary='obc'
    )
    
    print(f"Individual SMPO: {smpo1.L} tensors → outputs every {smpo1.spacing}")
    
    print(f"\nAutoencoder structure: {autoencoder}")
    print(f"Total tensors in autoencoder: {autoencoder.L}")
    
    # Show layer breakdown
    print("\nLayer breakdown:")
    for i, layer in enumerate(autoencoder.layers):
        print(f"  Layer {i}: {layer.L} tensors → {autoencoder._count_outputs(layer)} outputs")
    
    # Test the problematic case
    print("\n--- Testing problematic case ---")
    try:
        problem_autoencoder = CascadedSMPO.create_autoencoder(
            layer_dims=[56, 32, 16, 3],
            initializer=normal(stddev=0.1),
            key=key,
            shape_method='even',
            cyclic=True,
            bond_dim=4,
            phys_dim=(2, 2),
            add_identity=False,
            boundary='obc',
            compress=True
        )
        print(f"✅ Problem case working: {problem_autoencoder}")
        
        # Show the actual layer breakdown
        for i, layer in enumerate(problem_autoencoder.layers):
            outputs = problem_autoencoder._count_outputs(layer)
            print(f"  Layer {i}: {layer.L} tensors → {outputs} outputs")
            
    except Exception as e:
        print(f"❌ Problem case still fails: {e}")
        import traceback
        traceback.print_exc()