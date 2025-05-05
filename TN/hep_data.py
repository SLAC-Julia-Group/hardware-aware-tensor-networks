import h5py
import tensorflow as tf
import numpy as np

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

def prepare_training_data_from_structured(structured_dataset, max_batches=None, dtype=np.float64):
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

def h5_to_data_format(h5_file_path, dataset_key="Particles", batch_size=64, dtype=np.float64, max_batches=None):
    """
    Load H5 file data and convert directly to training format with minimal memory overhead.
    
    Args:
        h5_file_path: Path to the H5 file
        dataset_key: Key of the dataset in the H5 file
        batch_size: Size of the batches for processing
        dtype: Data type for the output array
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        Numpy array with shape (n_samples, 56) ready for tensor network training
    """
    # Open H5 file
    h5_file = h5py.File(h5_file_path, 'r')
    
    # Create TensorFlow dataset
    from hep_data import create_tf_dataset
    dataset = create_tf_dataset(h5_file, dataset_key, batch_size=batch_size)
    
    # Apply the transformation to structure the data
    def map_structure(batch):
        # Initialize list for features
        all_features = []
        
        # MET features (pt, phi)
        all_features.append(tf.transpose(batch[:, 0:1, 0], [1, 0]))    # pt
        all_features.append(tf.transpose(batch[:, 0:1, 2], [1, 0]))    # phi
        
        # Electron features (pt, eta, phi)
        all_features.append(tf.transpose(batch[:, 1:5, 0], [1, 0]))    # pt
        all_features.append(tf.transpose(batch[:, 1:5, 1], [1, 0]))    # eta
        all_features.append(tf.transpose(batch[:, 1:5, 2], [1, 0]))    # phi
        
        # Muon features (pt, eta, phi)
        all_features.append(tf.transpose(batch[:, 5:9, 0], [1, 0]))    # pt
        all_features.append(tf.transpose(batch[:, 5:9, 1], [1, 0]))    # eta
        all_features.append(tf.transpose(batch[:, 5:9, 2], [1, 0]))    # phi
        
        # Jet features (pt, eta, phi)
        all_features.append(tf.transpose(batch[:, 9:19, 0], [1, 0]))   # pt
        all_features.append(tf.transpose(batch[:, 9:19, 1], [1, 0]))   # eta
        all_features.append(tf.transpose(batch[:, 9:19, 2], [1, 0]))   # phi
        
        # Concatenate along first dimension (56, batch_size)
        concatenated = tf.concat(all_features, axis=0)
        
        # Transpose to get (batch_size, 56)
        return tf.transpose(concatenated)
    
    # Apply the transformation to the dataset
    processed_dataset = dataset.map(map_structure)
    
    # Determine total batches if needed
    if max_batches is None:
        max_batches = sum(1 for _ in processed_dataset)
    
    # Preallocate output array if batch size is consistent
    # Get the shape of the first batch to determine dimensions
    for first_batch in processed_dataset.take(1):
        batch_shape = first_batch.shape
        total_samples = min(max_batches * batch_shape[0], 
                            h5_file[dataset_key].shape[0])
        training_data = np.zeros((total_samples, batch_shape[1]), dtype=dtype)
        break
    
    # Fill the array batch by batch
    sample_idx = 0
    for i, batch in enumerate(processed_dataset):
        if i >= max_batches:
            break
            
        # Get batch as numpy array
        batch_np = batch.numpy()
        batch_size = batch_np.shape[0]
        
        # Add to the preallocated array
        training_data[sample_idx:sample_idx+batch_size] = batch_np
        sample_idx += batch_size
    
    # Resize if we didn't use all preallocated space
    if sample_idx < total_samples:
        training_data = training_data[:sample_idx]
    
    # Close the H5 file
    h5_file.close()
    
    return training_data