import h5py
# import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional

def print_keys(h5_file):
    print("Keys in the h5 file:", list(h5_file.keys()))
    # For each key, print basic information
    for key in h5_file.keys():
        item = h5_file[key]
        if isinstance(item, h5py.Dataset):  # If it's a dataset
            print(f"Dataset '{key}': shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):  # If it's a group
            print(f"Group '{key}' containing: {list(item.keys())}")

def print_info_for_key(h5_file):
    # Print a small sample of data if it's a dataset
    for key in h5_file.keys():
        if isinstance(h5_file[key], h5py.Dataset) and len(h5_file[key].shape) > 0:
            print(f"\nSample from dataset '{key}':")
            sample = h5_file[key][:5]  # First 5 items
            print(sample)

def create_tf_dataset(h5_file, dataset_key, batch_size=32, shuffle=True, buffer_size=1000):
    """
    Creates a TensorFlow dataset from an h5 dataset for batched processing.
    
    Args:
        h5_file: The h5py File object
        dataset_key: The key of the dataset to use
        batch_size: Size of the batches
        shuffle: Whether to shuffle the data
        buffer_size: Buffer size for shuffling
        
    Returns:
        A TensorFlow dataset that yields batches of data
    """
    # Get the dataset from the h5 file
    data = h5_file[dataset_key][:]
    
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def structure_physics_data(dataset):
    """
    Structure the raw batched data into a hierarchical format by particle type and feature.
    
    Args:
        dataset: TensorFlow dataset with batched data
        
    Returns:
        TensorFlow dataset with structured batches
    """
    structured_dataset = dataset.map(lambda batch: {
        'MET': {
            'pt': tf.transpose(batch[:, 0:1, 0], [1, 0]),    # Shape: (1, batch_size)
            'phi': tf.transpose(batch[:, 0:1, 2], [1, 0]),   # Shape: (1, batch_size)
        },
        'Ele': {
            'pt': tf.transpose(batch[:, 1:5, 0], [1, 0]),    # Shape: (4, batch_size)
            'eta': tf.transpose(batch[:, 1:5, 1], [1, 0]),   # Shape: (4, batch_size)
            'phi': tf.transpose(batch[:, 1:5, 2], [1, 0]),   # Shape: (4, batch_size)
        },
        'Mu': {
            'pt': tf.transpose(batch[:, 5:9, 0], [1, 0]),    # Shape: (4, batch_size)
            'eta': tf.transpose(batch[:, 5:9, 1], [1, 0]),   # Shape: (4, batch_size)
            'phi': tf.transpose(batch[:, 5:9, 2], [1, 0]),   # Shape: (4, batch_size)
        },
        'Jet': {
            'pt': tf.transpose(batch[:, 9:19, 0], [1, 0]),   # Shape: (10, batch_size)
            'eta': tf.transpose(batch[:, 9:19, 1], [1, 0]),  # Shape: (10, batch_size)
            'phi': tf.transpose(batch[:, 9:19, 2], [1, 0]),  # Shape: (10, batch_size)
        }
    })
    
    return structured_dataset

def inspect_structured_data(structured_dataset, num_batches=1):
    """
    Print information about the structured dataset for inspection.
    
    Args:
        structured_dataset: Structured TensorFlow dataset
        num_batches: Number of batches to inspect
    """
    for i, batch_dict in enumerate(structured_dataset.take(num_batches)):
        print(f"Batch {i+1}:")
        print("MET pt shape:", batch_dict['MET']['pt'].shape)
        print("Ele eta shape:", batch_dict['Ele']['eta'].shape)
        print("Mu phi shape:", batch_dict['Mu']['phi'].shape)
        print("Jet pt shape:", batch_dict['Jet']['pt'].shape)
        
        print("\nLeading MET pt:")
        print(batch_dict['MET']['pt'][0])  # First MET across batch
        
        print("\nLeading Electron pt:")
        print(batch_dict['Ele']['pt'][0])  # First electron pt across batch
        
        print("\nLeading Muon phi:")
        print(batch_dict['Mu']['phi'][0])  # First muon phi across batch
        
        print("\nLeading Jet pt:")
        print(batch_dict['Jet']['pt'][0])  # First jet pt across batch

def plot_physics_features(structured_dataset, output_dir="feature_plots", max_batches=None):
    """
    Plot distributions of physics features from the structured dataset.
    
    Args:
        structured_dataset: Structured TensorFlow dataset
        output_dir: Directory to save plots
        max_batches: Maximum number of batches to process (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine total batches if needed
    if max_batches is None:
        max_batches = sum(1 for _ in structured_dataset)
    
    # Collections for storing data
    data_collections = {
        'MET': {'pt': [], 'phi': []},
        'Ele': {'pt': [], 'eta': [], 'phi': []},
        'Mu': {'pt': [], 'eta': [], 'phi': []},
        'Jet': {'pt': [], 'eta': [], 'phi': []}
    }
    
    # Process each batch
    for i, batch_dict in enumerate(structured_dataset):
        if i >= max_batches:
            break
            
        # Collect data for each particle type and feature
        # MET features
        for feature in ['pt', 'phi']:
            data_collections['MET'][feature].extend(batch_dict['MET'][feature].numpy().flatten())
        
        # Electron features
        for feature in ['pt', 'eta', 'phi']:
            data_collections['Ele'][feature].extend(batch_dict['Ele'][feature].numpy().flatten())
        
        # Muon features
        for feature in ['pt', 'eta', 'phi']:
            data_collections['Mu'][feature].extend(batch_dict['Mu'][feature].numpy().flatten())
        
        # Jet features
        for feature in ['pt', 'eta', 'phi']:
            data_collections['Jet'][feature].extend(batch_dict['Jet'][feature].numpy().flatten())
    
    # Convert lists to numpy arrays
    for particle in data_collections:
        for feature in data_collections[particle]:
            data_collections[particle][feature] = np.array(data_collections[particle][feature])
    
    # Plot styling
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'MET': 'crimson', 'Ele': 'royalblue', 'Mu': 'forestgreen', 'Jet': 'darkorange'}
    labels = {
        'pt': 'p$_T$ [GeV]', 
        'eta': '$\eta$', 
        'phi': '$\phi$ [rad]'
    }
    
    # Create plots for each feature
    for particle in data_collections:
        for feature in data_collections[particle]:
            values = data_collections[particle][feature]
            
            plt.figure(figsize=(10, 6))
            plt.yscale('log')
            plt.ylim(1e-3, 10)  # Set y-axis range
    
            if feature == 'pt':
                counts, bins = np.histogram(values, bins=100, range=(0, 100))
                counts = counts / counts.sum()  # Normalize
                plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors[particle], alpha=0.7)
                plt.xticks(np.arange(0, 101, 10))
            elif feature == 'eta':
                counts, bins = np.histogram(values, bins=101, range=(-5, 5))
                counts = counts / counts.sum()  # Normalize
                plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors[particle], alpha=0.7)
                plt.xticks(np.arange(-5, 5.1, 1))
            elif feature == 'phi':
                counts, bins = np.histogram(values, bins=101, range=(-np.pi, np.pi))
                counts = counts / counts.sum()  # Normalize
                plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors[particle], alpha=0.7)
                plt.xticks(np.linspace(-np.pi, np.pi, 13), 
                          ["-π", "", "-2π/3", "", "-π/3", "", "0", "", "π/3", "", "2π/3", "", "π"])
    
            plt.title(f'{particle} {feature} Distribution', fontsize=16)
            plt.xlabel(labels[feature], fontsize=14)
            plt.ylabel('Events', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            filename = f"{output_dir}/{particle}_{feature}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            
            print(f"Saved plot to {filename}")
    
    print(f"\nAll plots saved to '{output_dir}' directory")

def prepare_training_data_from_structured(structured_dataset, max_batches=None, dtype=np.float32):
    """
    Convert structured dataset to a format suitable for tensor network training.
    
    Args:
        structured_dataset: Structured TensorFlow dataset
        max_batches: Maximum number of batches to process (None for all)
        dtype: Data type for the output array
        
    Returns:
        Numpy array with shape (num_samples, 56) ready for model training
    """
    # Determine total batches if needed
    if max_batches is None:
        max_batches = sum(1 for _ in structured_dataset)
    
    all_batches = []
    
    # Process each batch
    for i, batch_dict in enumerate(structured_dataset):
        if i >= max_batches:
            break
        
        # Initialize list for features in this batch
        all_features = []
        
        # Add MET features
        all_features.append(batch_dict['MET']['pt'])
        all_features.append(batch_dict['MET']['phi'])
        
        # Add Electron features
        all_features.append(batch_dict['Ele']['pt'])
        all_features.append(batch_dict['Ele']['eta'])
        all_features.append(batch_dict['Ele']['phi'])
        
        # Add Muon features
        all_features.append(batch_dict['Mu']['pt'])
        all_features.append(batch_dict['Mu']['eta'])
        all_features.append(batch_dict['Mu']['phi'])
        
        # Add Jet features
        all_features.append(batch_dict['Jet']['pt'])
        all_features.append(batch_dict['Jet']['eta'])
        all_features.append(batch_dict['Jet']['phi'])
        
        # Concatenate features (56, batch_size)
        concatenated = tf.concat(all_features, axis=0)
        
        # Transpose to (batch_size, 56)
        train_format = tf.transpose(concatenated)
        
        # Convert to numpy and append
        all_batches.append(train_format.numpy())
    
    # Concatenate all batches
    train_data = np.concatenate(all_batches, axis=0)
    
    # Ensure correct data type
    return train_data.astype(dtype)

def h5_to_jax_array(h5_file_path, dataset_key="Particles", dtype=np.float32):
    """
    Load H5 file data and convert directly to JAX array format with minimal memory overhead.
    
    Args:
        h5_file_path: Path to the H5 file
        dataset_key: Key of the dataset in the H5 file
        dtype: Data type for the output array
        
    Returns:
        JAX array with shape (n_samples, 56) ready for tensor network training
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        dataset = h5_file[dataset_key]
        n_samples = dataset.shape[0]
        
        # Preallocate output array to avoid intermediate copies
        output = np.zeros((n_samples, 56), dtype=dtype)
        
        # MET features (indices 0-1)
        output[:, 0] = dataset[:, 0, 0]  # pt
        output[:, 1] = dataset[:, 0, 2]  # phi
        
        # Electron features (indices 2-13)
        for e in range(4):
            output[:, 2+e] = dataset[:, 1+e, 0]    # e pt
            output[:, 6+e] = dataset[:, 1+e, 1]    # e eta
            output[:, 10+e] = dataset[:, 1+e, 2]   # e phi
        
        # Muon features (indices 14-25)
        for m in range(4):
            output[:, 14+m] = dataset[:, 5+m, 0]   # mu pt
            output[:, 18+m] = dataset[:, 5+m, 1]   # mu eta
            output[:, 22+m] = dataset[:, 5+m, 2]   # mu phi
        
        # Jet features (indices 26-55)
        for j in range(10):
            output[:, 26+j] = dataset[:, 9+j, 0]   # jet pt
            output[:, 36+j] = dataset[:, 9+j, 1]   # jet eta
            output[:, 46+j] = dataset[:, 9+j, 2]   # jet phi
        
    # Convert to JAX array - should be a shallow operation if possible
    return jnp.asarray(output)

class LazyH5Array:
    def __init__(self, h5_file_path: str, dataset_key: str = "Particles", dtype=np.float32):
        """
        A lazy loading array that behaves like a JAX array but only loads data on demand.
        
        Args:
            h5_file_path: Path to the H5 file
            dataset_key: Key of the dataset in the H5 file
            dtype: Data type for the output array
        """
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key
        self.dtype = dtype
        
        # Open the file to get shape, but don't load data
        with h5py.File(h5_file_path, 'r') as h5_file:
            self.dataset_shape = h5_file[dataset_key].shape
            # Cache the output shape for quick access
            self.shape = (self.dataset_shape[0], 56)
        
        # Keep track of loaded chunks
        self._cache = {}
    
    def __len__(self):
        """Return the number of samples."""
        return self.shape[0]
    
    def _process_chunk(self, data_chunk):
        """Transform a chunk of the raw data to the desired output format."""
        n_samples = data_chunk.shape[0]
        output = np.zeros((n_samples, 56), dtype=self.dtype)
        
        # MET features (indices 0-1)
        output[:, 0] = data_chunk[:, 0, 0]  # pt
        output[:, 1] = data_chunk[:, 0, 2]  # phi
        
        # Electron features (indices 2-13)
        for e in range(4):
            output[:, 2+e] = data_chunk[:, 1+e, 0]    # e pt
            output[:, 6+e] = data_chunk[:, 1+e, 1]    # e eta
            output[:, 10+e] = data_chunk[:, 1+e, 2]   # e phi
        
        # Muon features (indices 14-25)
        for m in range(4):
            output[:, 14+m] = data_chunk[:, 5+m, 0]   # mu pt
            output[:, 18+m] = data_chunk[:, 5+m, 1]   # mu eta
            output[:, 22+m] = data_chunk[:, 5+m, 2]   # mu phi
        
        # Jet features (indices 26-55)
        for j in range(10):
            output[:, 26+j] = data_chunk[:, 9+j, 0]   # jet pt
            output[:, 36+j] = data_chunk[:, 9+j, 1]   # jet eta
            output[:, 46+j] = data_chunk[:, 9+j, 2]   # jet phi
        
        return output
    
    def _get_chunk(self, start_idx, end_idx):
        """Load a chunk from the H5 file if not in cache."""
        chunk_key = (start_idx, end_idx)
        if chunk_key not in self._cache:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                raw_chunk = h5_file[self.dataset_key][start_idx:end_idx]
                self._cache[chunk_key] = self._process_chunk(raw_chunk)
        return self._cache[chunk_key]
    
    def __getitem__(self, idx):
        """Support array-like indexing but load data on demand."""
        if isinstance(idx, int):
            # Single item access
            chunk = self._get_chunk(idx, idx+1)
            return jnp.asarray(chunk[0])
        elif isinstance(idx, slice):
            # Slice access
            start = idx.start or 0
            stop = idx.stop or len(self)
            step = idx.step or 1
            
            if step != 1:
                # For non-unit steps, we need to load separate chunks
                indices = range(start, stop, step)
                result = np.zeros((len(indices), 56), dtype=self.dtype)
                for i, idx in enumerate(indices):
                    result[i] = self._get_chunk(idx, idx+1)[0]
                return jnp.asarray(result)
            else:
                # For unit steps, we can load the whole range at once
                return jnp.asarray(self._get_chunk(start, stop))
        elif isinstance(idx, tuple) and len(idx) == 2:
            # 2D indexing
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                chunk = self._get_chunk(idx[0], idx[0]+1)
                return jnp.asarray(chunk[0, idx[1]])
            else:
                # Handle more complex slicing
                # (simplified implementation - would need to handle all cases)
                row_slice = idx[0]
                col_slice = idx[1]
                if isinstance(row_slice, slice):
                    start = row_slice.start or 0
                    stop = row_slice.stop or len(self)
                    chunk = self._get_chunk(start, stop)
                    return jnp.asarray(chunk[:, col_slice])
                else:
                    raise NotImplementedError("Complex indexing not fully implemented")
        else:
            raise IndexError("Unsupported indexing")