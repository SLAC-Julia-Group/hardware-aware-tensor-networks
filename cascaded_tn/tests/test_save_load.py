"""
Comprehensive unit tests for CascadedModel save/load functionality.

Tests verify that saved and loaded models are identical in:
- Architecture configuration
- Trained weights (every element)
- ReLU settings
- All hyperparameters
"""

import pytest
import tempfile
import os
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascaded_tn.training.cascaded_model import CascadedModel, create_trainable_encoder


class TestSaveLoad:
    """Test suite for model save/load functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return create_trainable_encoder(
            layer_dims=[10, 5, 3],
            bond_dims=[4, 3],
            phys_dims=[2, 2, 2],
            enable_relu=[0],  # ReLU on first layer
            cyclic=False,
            key=jax.random.PRNGKey(42),
            debug=False,
            initializer=jax.nn.initializers.normal(stddev=0.1),
        )

    @pytest.fixture
    def complex_model(self):
        """Create a complex model with multiple ReLU layers."""
        return create_trainable_encoder(
            layer_dims=[19, 7, 3],
            bond_dims=[8, 3],
            phys_dims=[3, 2, 3],
            enable_relu=[0, 1],  # ReLU on both layers
            cyclic=False,
            key=jax.random.PRNGKey(123),
            debug=False,
            initializer=jax.nn.initializers.variance_scaling(scale=2.0, mode='fan_avg', distribution='uniform'),
            add_identity=True,
        )
    
    def modify_weights(self, model):
        """Modify model weights to simulate training."""
        # Get current arrays
        arrays = model.arrays
        
        # Modify them with a predictable pattern
        modified_arrays = []
        for i, arr in enumerate(arrays):
            # Add a small offset based on tensor index
            offset = (i + 1) * 0.1
            modified = arr + offset
            modified_arrays.append(modified)
        
        # Update model
        model.update_tensors(modified_arrays)
        
        return modified_arrays
    
    def test_basic_save_load(self, simple_model, temp_dir):
        """Test basic save and load functionality."""
        filepath = os.path.join(temp_dir, 'test_model.pkl')
        
        # Save model
        simple_model.save(filepath)
        assert os.path.exists(filepath), "Save file was not created"
        
        # Load model
        loaded_model = CascadedModel.load(filepath)
        assert loaded_model is not None, "Failed to load model"
        
        # Basic checks
        assert loaded_model.L == simple_model.L, "Number of tensors mismatch"
        assert loaded_model.nparams() == simple_model.nparams(), "Parameter count mismatch"
    
    def test_architecture_preservation(self, simple_model, temp_dir):
        """Test that architecture is preserved exactly."""
        filepath = os.path.join(temp_dir, 'test_arch.pkl')
        
        # Extract original config
        original_config = simple_model._extract_config()
        
        # Save and load
        simple_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        loaded_config = loaded_model._extract_config()
        
        # Compare configurations
        assert original_config['layer_dims'] == loaded_config['layer_dims'], \
            f"Layer dims mismatch: {original_config['layer_dims']} != {loaded_config['layer_dims']}"
        
        assert original_config['bond_dims'] == loaded_config['bond_dims'], \
            f"Bond dims mismatch: {original_config['bond_dims']} != {loaded_config['bond_dims']}"
        
        assert original_config['phys_dims'] == loaded_config['phys_dims'], \
            f"Phys dims mismatch: {original_config['phys_dims']} != {loaded_config['phys_dims']}"
        
        assert original_config['enable_relu'] == loaded_config['enable_relu'], \
            f"ReLU config mismatch: {original_config['enable_relu']} != {loaded_config['enable_relu']}"
        
        assert original_config['cyclic'] == loaded_config['cyclic'], \
            f"Cyclic flag mismatch"
    
    def test_relu_configuration(self, complex_model, temp_dir):
        """Test that ReLU configuration is preserved correctly."""
        filepath = os.path.join(temp_dir, 'test_relu.pkl')
        
        # Check original ReLU settings
        original_relu = [op.config.enable_relu for op in complex_model.cascade.operators]
        assert original_relu == [True, True], f"Expected [True, True], got {original_relu}"
        
        # Save and load
        complex_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        
        # Check loaded ReLU settings
        loaded_relu = [op.config.enable_relu for op in loaded_model.cascade.operators]
        assert loaded_relu == original_relu, \
            f"ReLU config mismatch: original={original_relu}, loaded={loaded_relu}"
    
    def test_weight_preservation_exact(self, simple_model, temp_dir):
        """Test that every single weight element is preserved exactly."""
        filepath = os.path.join(temp_dir, 'test_weights.pkl')
        
        # Modify weights to create unique pattern
        original_arrays = self.modify_weights(simple_model)
        
        # Save and load
        simple_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        
        # Get loaded arrays
        loaded_arrays = loaded_model.arrays
        
        # Check number of arrays
        assert len(loaded_arrays) == len(original_arrays), \
            f"Number of arrays mismatch: {len(loaded_arrays)} != {len(original_arrays)}"
        
        # Compare each array element-by-element
        for i, (orig, loaded) in enumerate(zip(original_arrays, loaded_arrays)):
            # Check shapes match
            assert orig.shape == loaded.shape, \
                f"Tensor {i} shape mismatch: {orig.shape} != {loaded.shape}"
            
            # Convert to numpy for comparison
            orig_np = np.array(jax.device_get(orig))
            loaded_np = np.array(jax.device_get(loaded))
            
            # Check every element matches exactly
            np.testing.assert_array_equal(
                orig_np, loaded_np,
                err_msg=f"Tensor {i} has different values"
            )
            
            # Also check with allclose for floating point
            np.testing.assert_allclose(
                orig_np, loaded_np,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Tensor {i} values differ beyond tolerance"
            )
    
    def test_physical_dimensions_all_layers(self, complex_model, temp_dir):
        """Test that physical dimensions are preserved for all layers."""
        filepath = os.path.join(temp_dir, 'test_phys_dims.pkl')
        
        # Get original physical dimensions
        original_phys_dims = [op.config.phys_dim for op in complex_model.cascade.operators]
        
        # Save and load
        complex_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        
        # Get loaded physical dimensions
        loaded_phys_dims = [op.config.phys_dim for op in loaded_model.cascade.operators]
        
        # Compare each layer
        assert len(loaded_phys_dims) == len(original_phys_dims), \
            f"Number of layers mismatch"
        
        for i, (orig_phys, loaded_phys) in enumerate(zip(original_phys_dims, loaded_phys_dims)):
            assert orig_phys == loaded_phys, \
                f"Layer {i} phys_dim mismatch: {orig_phys} != {loaded_phys}"
    
    def test_add_identity_flag(self, complex_model, temp_dir):
        """Test that add_identity flag is preserved."""
        filepath = os.path.join(temp_dir, 'test_identity.pkl')
        
        # Verify add_identity is True (as set in fixture)
        original_add_identity = complex_model.cascade.operators[0].config.add_identity
        assert original_add_identity == True, "add_identity should be True in complex_model"
        
        # Save and load
        complex_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        
        # Check loaded model
        loaded_add_identity = loaded_model.cascade.operators[0].config.add_identity
        assert loaded_add_identity == original_add_identity, \
            f"add_identity flag mismatch: {loaded_add_identity} != {original_add_identity}"
    
    def test_file_extension_handling(self, simple_model, temp_dir):
        """Test that .pkl extension is added automatically."""
        # Save without extension
        filepath_no_ext = os.path.join(temp_dir, 'test_model')
        simple_model.save(filepath_no_ext)
        
        # Should create .pkl file
        expected_path = filepath_no_ext + '.pkl'
        assert os.path.exists(expected_path), "Should create .pkl file automatically"
        
        # Should be able to load with or without extension
        loaded1 = CascadedModel.load(filepath_no_ext)
        loaded2 = CascadedModel.load(expected_path)
        
        assert loaded1.nparams() == loaded2.nparams()
    
    def test_metadata_preservation(self, simple_model, temp_dir):
        """Test that metadata is saved and can be read."""
        filepath = os.path.join(temp_dir, 'test_metadata.pkl')
        
        # Get original metadata
        original_n_params = simple_model.nparams()
        original_n_layers = len(simple_model.cascade.operators)
        original_n_tensors = simple_model.L
        
        # Save
        simple_model.save(filepath)
        
        # Load pickle directly to check metadata
        import pickle
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        metadata = save_dict['metadata']
        
        assert metadata['model_type'] == 'CascadedModel'
        assert metadata['n_params'] == original_n_params
        assert metadata['n_layers'] == original_n_layers
        assert metadata['n_tensors'] == original_n_tensors
    
    def test_modified_weights_different_from_original(self, simple_model, temp_dir):
        """Sanity check: modified weights should actually be different."""
        # Get original weights
        original_arrays = [np.array(jax.device_get(arr)) for arr in simple_model.arrays]
        
        # Modify weights
        modified_arrays = self.modify_weights(simple_model)
        modified_np = [np.array(jax.device_get(arr)) for arr in modified_arrays]
        
        # Verify they're different
        for i, (orig, mod) in enumerate(zip(original_arrays, modified_np)):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(orig, mod)
    
    def test_multiple_save_load_cycles(self, simple_model, temp_dir):
        """Test that multiple save/load cycles preserve the model."""
        # Modify weights
        self.modify_weights(simple_model)
        original_arrays = [np.array(jax.device_get(arr)) for arr in simple_model.arrays]
        
        # Save/load cycle 1
        filepath1 = os.path.join(temp_dir, 'cycle1.pkl')
        simple_model.save(filepath1)
        loaded1 = CascadedModel.load(filepath1)
        
        # Save/load cycle 2
        filepath2 = os.path.join(temp_dir, 'cycle2.pkl')
        loaded1.save(filepath2)
        loaded2 = CascadedModel.load(filepath2)
        
        # Save/load cycle 3
        filepath3 = os.path.join(temp_dir, 'cycle3.pkl')
        loaded2.save(filepath3)
        loaded3 = CascadedModel.load(filepath3)
        
        # Compare final with original
        final_arrays = [np.array(jax.device_get(arr)) for arr in loaded3.arrays]
        
        for i, (orig, final) in enumerate(zip(original_arrays, final_arrays)):
            np.testing.assert_allclose(
                orig, final,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Tensor {i} differs after multiple save/load cycles"
            )
    
    def test_all_operators_have_correct_config(self, complex_model, temp_dir):
        """Test that all operators in the cascade maintain their configuration."""
        filepath = os.path.join(temp_dir, 'test_operators.pkl')
        
        # Get original operator configs
        original_configs = [
            {
                'input_dim': op.config.input_dim,
                'output_dim': op.config.output_dim,
                'bond_dim': op.config.bond_dim,
                'cyclic': op.config.cyclic,
                'phys_dim': op.config.phys_dim,
                'add_identity': op.config.add_identity,
                'enable_relu': op.config.enable_relu,
            }
            for op in complex_model.cascade.operators
        ]
        
        # Save and load
        complex_model.save(filepath)
        loaded_model = CascadedModel.load(filepath)
        
        # Get loaded operator configs
        loaded_configs = [
            {
                'input_dim': op.config.input_dim,
                'output_dim': op.config.output_dim,
                'bond_dim': op.config.bond_dim,
                'cyclic': op.config.cyclic,
                'phys_dim': op.config.phys_dim,
                'add_identity': op.config.add_identity,
                'enable_relu': op.config.enable_relu,
            }
            for op in loaded_model.cascade.operators
        ]
        
        # Compare each operator
        assert len(loaded_configs) == len(original_configs), "Number of operators mismatch"
        
        for i, (orig, loaded) in enumerate(zip(original_configs, loaded_configs)):
            for key in orig.keys():
                assert orig[key] == loaded[key], \
                    f"Operator {i} config mismatch for '{key}': {orig[key]} != {loaded[key]}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])