import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import NamedTuple
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
    def __init__(self, file_path, key="Particles", norm=False, skip_MET_eta=True):
        self.file_path = file_path       
        self.file = h5py.File(self.file_path, 'r')

        self.ds = self.file[key]

        self.empty_array =  np.empty(self.file[key].shape, dtype=self.file[key].dtype)

        self.stats = None # mean and var
        self.norm = norm
        self.skip_MET_eta = skip_MET_eta
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

    

def load_data(file_path, batch_size=64, shuffle=True, num_workers=0, drop_last=False, norm=False, skip_MET_eta=True):
    dataset = HDF5Dataset(file_path, norm=norm, skip_MET_eta=skip_MET_eta)
    loader = DataLoader(dataset, 
                        batch_size=None, 
                        shuffle=False,  
                        sampler=RandomBatchSampler(dataset, batch_size, shuffle, drop_last),
                        num_workers=num_workers, 
                        collate_fn=None,
                        pin_memory=True # for faster gpu transfer
                       )

    return loader