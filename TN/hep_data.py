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
