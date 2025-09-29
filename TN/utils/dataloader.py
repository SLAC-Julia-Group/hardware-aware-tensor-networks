import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import NamedTuple, Tuple, Optional
from utils.dataset import var_dict, get_norm_dict
import os

import time

from torch.utils.data import Sampler


class RandomBatchSampler(Sampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Batch sampler for an h5 dataset.

        The batch sampler performs weak shuffling. Objects are batched first,
        and then batches are shuffled.

        Parameters
        ----------
        dataset : torch.data.Dataset
            Input dataset
        batch_size : int
            Number of objects to batch
        shuffle : bool
            Shuffle the batches
        drop_last : bool
            Drop the last incomplete batch (if present)
        """
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.nonzero_last_batch = int(self.n_batches) < self.n_batches
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        return int(self.n_batches) + int(not self.drop_last and self.nonzero_last_batch)

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(int(self.n_batches))
        else:
            self.batch_ids = torch.arange(int(self.n_batches))
        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start, stop = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            yield np.s_[int(start) : int(stop)]

        # in case the batch size is not a perfect multiple of the number of samples,
        # yield the remaining samples
        if not self.drop_last and self.nonzero_last_batch:
            start, stop = int(self.n_batches) * self.batch_size, self.dataset_length
            yield np.s_[int(start) : int(stop)]


class HDF5Dataset(Dataset):
    def __init__(self, file_path, key="Particles", norm=False, skip_MET_eta=True, cartesian=False):
        self.file_path = file_path       
        self.file = h5py.File(self.file_path, 'r')

        self.ds = self.file[key]

        self.empty_array =  np.empty(self.file[key].shape, dtype=self.file[key].dtype)

        self.stats = None # mean and var
        self.norm = norm
        self.skip_MET_eta = skip_MET_eta
        self.cartesian = cartesian
        self.n_features = 56 if skip_MET_eta else 57

    def __len__(self):
        # n_jets dimension
        return self.ds.shape[0]
    
    def __getitem__(self, object_idx):
        """Return on sample or batch from the dataset.

        Parameters
        ----------
        object_idx
            A numpy slice corresponding to a batch of objects.

        Returns
        -------
        tuple
            Dict of tensor for each of the inputs, pad_masks, and labels.
            Each tensor will contain a batch of samples.
        """                   

        batch = self.empty_array
        shape = (object_idx.stop - object_idx.start,) + self.ds.shape[1:]
        batch.resize(shape, refcheck=False)
        self.ds.read_direct(batch, object_idx) # load data to batch
    
        B = batch.shape[0]
        features = np.empty((B, self.n_features), dtype=batch.dtype)

        # Extract features grouped by particle
        idx = 0
        
        # MET (1 particle)
        features[:, idx] = batch[:, 0, 0]  # MET pt
        idx += 1
        if not self.skip_MET_eta:
            features[:, idx] = batch[:, 0, 1]  # MET eta
            idx += 1
        features[:, idx] = batch[:, 0, 2]  # MET phi
        idx += 1
        
        # Electrons (4 particles)
        for i in range(1, 5):
            features[:, idx] = batch[:, i, 0]  # pt
            features[:, idx+1] = batch[:, i, 1]  # eta
            features[:, idx+2] = batch[:, i, 2]  # phi
            idx += 3
        
        # Muons (4 particles)
        for i in range(5, 9):
            features[:, idx] = batch[:, i, 0]  # pt
            features[:, idx+1] = batch[:, i, 1]  # eta
            features[:, idx+2] = batch[:, i, 2]  # phi
            idx += 3
        
        # Jets (10 particles)
        for i in range(9, 19):
            features[:, idx] = batch[:, i, 0]  # pt
            features[:, idx+1] = batch[:, i, 1]  # eta
            features[:, idx+2] = batch[:, i, 2]  # phi
            idx += 3

        # Convert from (pt, eta, phi) to (px, py, pz)
        if self.cartesian:
            # features_cart = np.empty((B, self.n_features), dtype=batch.dtype)
            # baseline = 0. #GeV
            idx = 0

            # MET
            pt = features[:, idx].copy()
            phi = features[:, idx+2].copy() if not self.skip_MET_eta else features[:, idx+1].copy()
            features[:, idx] = pt * np.cos(phi)  # px
            features[:, idx+1] = pt * np.sin(phi)  # py
            if not self.skip_MET_eta:
                features[:, idx+2] = 0  # pz
                idx += 3
            else:
                idx += 2

            # Electrons, Muons, Jets
            for _ in range(18):  # 4 electrons + 4 muons + 10 jets
                pt, eta, phi = features[:, idx].copy(), features[:, idx+1].copy(), features[:, idx+2].copy()
                mask = pt != 0
                features[:, idx] = np.where(mask, pt * np.cos(phi), 0)   # px
                features[:, idx+1] = np.where(mask, pt * np.sin(phi), 0)  # py
                features[:, idx+2] = np.where(mask, pt * np.sinh(np.clip(eta, -5, 5)), 0)  # pz
                idx += 3

        # Apply statistics-based normalization if available
        if self.stats:
            feature_names = self._get_ordered_feature_names()
            for i, name in enumerate(feature_names):
                if name in self.stats:
                    mu = self.stats[name]['mean']
                    std = np.sqrt(self.stats[name]['variance']) + 1e-6
                    features[:, i] = (features[:, i] - mu) / std
                    if np.any(np.isnan(features[:, i])): 
                        print(f"warning! {name}: mu={mu}, std={std} caused some nans")

        # Physics-aware normalization
        if self.norm:
            if not self.cartesian:
                idx = 0
                
                # MET
                features[:, idx] = features[:, idx] / 1200  # MET pt
                idx += 1
                if not self.skip_MET_eta:
                    features[:, idx] = (features[:, idx] + 5) / 10  # MET eta
                    idx += 1
                features[:, idx] = (features[:, idx] + np.pi) / (2 * np.pi)  # MET phi
                idx += 1
                
                # Electrons
                for i in range(4):
                    features[:, idx] = features[:, idx] / 1200  # pt
                    features[:, idx+1] = (features[:, idx+1] + 5) / 10  # eta
                    features[:, idx+2] = (features[:, idx+2] + np.pi) / (2 * np.pi)  # phi
                    idx += 3
                
                # Muons
                for i in range(4):
                    features[:, idx] = features[:, idx] / 800  # pt
                    features[:, idx+1] = (features[:, idx+1] + 5) / 10  # eta
                    features[:, idx+2] = (features[:, idx+2] + np.pi) / (2 * np.pi)  # phi
                    idx += 3
                
                # Jets
                for i in range(10):
                    features[:, idx] = features[:, idx] / 2500  # pt
                    features[:, idx+1] = (features[:, idx+1] + 5) / 10  # eta
                    features[:, idx+2] = (features[:, idx+2] + np.pi) / (2 * np.pi)  # phi
                    idx += 3

            else:
                NULL_UNIT = 1.0 / np.sqrt(3)
                idx = 0

                # MET (px, py only - no pz)
                features[:, idx] = features[:, idx] / 10  # MET px
                features[:, idx+1] = features[:, idx+1] / 10  # MET py
                idx += 2
                if not self.skip_MET_eta:
                    # MET has no pz, already 0 or padded
                    idx += 1

                # Electrons (4 particles)
                for i in range(4):
                    mask = features[:, idx] != 0
                    features[:, idx] = np.where(mask, features[:, idx] / 10, NULL_UNIT)  # px
                    features[:, idx+1] = np.where(mask, features[:, idx+1] / 10, NULL_UNIT)  # py
                    features[:, idx+2] = np.where(mask, features[:, idx+2] / 20, NULL_UNIT)  # pz
                    idx += 3
                
                # Muons (4 particles)
                for i in range(4):
                    mask = features[:, idx] != 0
                    features[:, idx] = np.where(mask, features[:, idx] / 10, NULL_UNIT)  # px
                    features[:, idx+1] = np.where(mask, features[:, idx+1] / 10, NULL_UNIT)  # py
                    features[:, idx+2] = np.where(mask, features[:, idx+2] / 20, NULL_UNIT)  # pz
                    idx += 3
                
                # Jets (10 particles)
                for i in range(10):
                    mask = features[:, idx] != 0
                    features[:, idx] = np.where(mask, features[:, idx] / 100, NULL_UNIT)  # px
                    features[:, idx+1] = np.where(mask, features[:, idx+1] / 100, NULL_UNIT)  # py
                    features[:, idx+2] = np.where(mask, features[:, idx+2] / 200, NULL_UNIT)  # pz
                    idx += 3

        return features

    def _get_ordered_feature_names(self):
        """Get feature names in the particle-grouped order."""
        names = []
        
        # MET
        names.append("met_pt")
        if not self.skip_MET_eta:
            names.append("met_eta")
        names.append("met_phi")
        
        # Electrons
        for i in range(4):
            names.extend([f"electron{i}_pt", f"electron{i}_eta", f"electron{i}_phi"])
        
        # Muons  
        for i in range(4):
            names.extend([f"mu{i}_pt", f"mu{i}_eta", f"mu{i}_phi"])
        
        # Jets
        for i in range(10):
            names.extend([f"jet{i}_pt", f"jet{i}_eta", f"jet{i}_phi"])
        
        return names

    def close(self):
        self.file.close()

class SplitHDF5Dataset(Dataset):
    """
    Wrapper for HDF5Dataset that provides train/val/test splits while maintaining lazy loading.
    
    This wrapper only handles index translation - all data processing happens in the
    underlying HDF5Dataset, preserving lazy loading behavior.
    """
    
    def __init__(self, base_dataset: HDF5Dataset, indices: np.ndarray):
        """
        Initialize split dataset wrapper.
        
        Parameters
        ----------
        base_dataset : HDF5Dataset
            The underlying HDF5 dataset with all data
        indices : np.ndarray
            Array of indices this split should use
        """
        self.base_dataset = base_dataset
        self.indices = indices
        
        # Pass through important attributes
        self.n_features = base_dataset.n_features
        self.norm = base_dataset.norm
        self.skip_MET_eta = base_dataset.skip_MET_eta
        self.cartesian = base_dataset.cartesian
        self.stats = base_dataset.stats
        
    def __len__(self):
        """Return the length of this split."""
        return len(self.indices)
    
    def __getitem__(self, local_idx):
        """
        Translate local index to global index and fetch from base dataset.
        
        Parameters
        ----------
        local_idx : np.s_ or int
            Local index or slice within this split
            
        Returns
        -------
        Data from the base dataset
        """
        # Handle both slice and integer indices
        if isinstance(local_idx, slice):
            # It's a numpy slice object (np.s_)
            start = local_idx.start if local_idx.start is not None else 0
            stop = local_idx.stop if local_idx.stop is not None else len(self.indices)
            
            # Get the global indices for this batch
            batch_indices = self.indices[start:stop]
            
            # Create a slice for the base dataset using min and max indices
            # This assumes the indices are sorted (which they will be for efficiency)
            global_start = batch_indices[0]
            global_stop = batch_indices[-1] + 1
            
            # For non-contiguous indices, we'd need to handle differently
            # For now, assume contiguous blocks (which is true for train/val/test splits)
            global_slice = np.s_[int(global_start):int(global_stop)]
            
            return self.base_dataset[global_slice]
        else:
            # Single index access
            global_idx = self.indices[local_idx]
            return self.base_dataset[global_idx]
    
    def close(self):
        """Close the underlying dataset."""
        self.base_dataset.close()


def create_dataset_splits(
    file_path: str,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    key: str = "Particles",
    norm: bool = False,
    skip_MET_eta: bool = True,
    cartesian: bool = False,
    seed: int = 42,
    shuffle_before_split: bool = False
) -> Tuple[SplitHDF5Dataset, SplitHDF5Dataset, SplitHDF5Dataset]:
    """
    Create train, validation, and test dataset splits from a single HDF5 file.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file
    split_ratios : tuple of float
        (train_ratio, val_ratio, test_ratio) - must sum to 1.0
    key : str
        Key for the dataset in the HDF5 file
    norm : bool
        Whether to apply normalization
    skip_MET_eta : bool
        Whether to skip MET eta feature
    cartesian : bool
        Whether to convert (pt, eta, phi) to (px, py, pz)
    seed : int
        Random seed for shuffling before split
    shuffle_before_split : bool
        Whether to shuffle indices before splitting
        
    Returns
    -------
    tuple of SplitHDF5Dataset
        (train_dataset, val_dataset, test_dataset)
    """
    # Validate split ratios
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # Create base dataset
    base_dataset = HDF5Dataset(file_path, key=key, norm=norm, skip_MET_eta=skip_MET_eta, cartesian=cartesian)
    
    # Get total number of samples
    n_samples = len(base_dataset)
    
    # Create indices
    indices = np.arange(n_samples)
    
    # Shuffle if requested
    if shuffle_before_split:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    
    # Calculate split points
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Sort indices within each split for better HDF5 access patterns
    # (sequential reads are much faster than random reads)
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)
    
    # Create split datasets
    train_dataset = SplitHDF5Dataset(base_dataset, train_indices)
    val_dataset = SplitHDF5Dataset(base_dataset, val_indices)
    test_dataset = SplitHDF5Dataset(base_dataset, test_indices)
    
    # Print split information
    print(f"[DATASET SPLITS]")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Train: {len(train_dataset):,} ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_dataset):,} ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_dataset):,} ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


def create_split_dataloaders(
    file_path: str,
    batch_size: int = 64,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    norm: bool = False,
    skip_MET_eta: bool = True,
    cartesian: bool = False,
    key: str = "Particles",
    seed: int = 42,
    shuffle_before_split: bool = False,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with automatic splitting.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file
    batch_size : int
        Batch size for training
    split_ratios : tuple of float
        (train_ratio, val_ratio, test_ratio) - must sum to 1.0
    shuffle : bool
        Whether to shuffle batches during training
    num_workers : int
        Number of workers for data loading
    drop_last : bool
        Whether to drop the last incomplete batch
    norm : bool
        Whether to apply normalization
    skip_MET_eta : bool
        Whether to skip MET eta feature
    cartesian : bool
        Whether to convert (pt, eta, phi) to (px, py, pz)
    key : str
        Key for the dataset in the HDF5 file
    seed : int
        Random seed for splitting
    shuffle_before_split : bool
        Whether to shuffle data before splitting
    val_batch_size : int or None
        Batch size for validation (defaults to batch_size)
    test_batch_size : int or None
        Batch size for testing (defaults to batch_size)
        
    Returns
    -------
    tuple of DataLoader
        (train_loader, val_loader, test_loader)
        
    Example
    -------
    >>> train_loader, val_loader, test_loader = create_split_dataloaders(
    ...     "background_for_training.h5",
    ...     batch_size=64,
    ...     split_ratios=(0.7, 0.15, 0.15),
    ...     norm=True,
    ...     skip_MET_eta=True
    ... )
    """
    # Use provided batch sizes or default to main batch_size
    val_batch_size = val_batch_size or batch_size
    test_batch_size = test_batch_size or batch_size
    
    # Create split datasets
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        file_path=file_path,
        split_ratios=split_ratios,
        key=key,
        norm=norm,
        skip_MET_eta=skip_MET_eta,
        cartesian=cartesian,
        seed=seed,
        shuffle_before_split=shuffle_before_split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        sampler=RandomBatchSampler(train_dataset, batch_size, shuffle, drop_last),
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        sampler=RandomBatchSampler(val_dataset, val_batch_size, shuffle=False, drop_last=False),
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        sampler=RandomBatchSampler(test_dataset, test_batch_size, shuffle=False, drop_last=False),
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True
    )
    
    print(f"\n[DATALOADERS CREATED]")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def load_data(file_path, batch_size=64, shuffle=True, num_workers=0, drop_last=False, norm=False, skip_MET_eta=True, cartesian=False):
    dataset = HDF5Dataset(file_path, norm=norm, skip_MET_eta=skip_MET_eta, cartesian=cartesian)
    loader = DataLoader(dataset, 
                        batch_size=None, 
                        shuffle=False,  
                        sampler=RandomBatchSampler(dataset, batch_size, shuffle, drop_last),
                        num_workers=num_workers, 
                        collate_fn=None,
                        pin_memory=True # for faster gpu transfer
                       )

    return loader