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
    def __init__(self, file_path, key="Particles", norm=False):
        self.file_path = file_path       
        self.file = h5py.File(self.file_path, 'r')

        self.ds = self.file[key]

        self.empty_array =  np.empty(self.file[key].shape, dtype=self.file[key].dtype)

        self.stats = None # mean and var
        self.norm = norm

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
        features = np.empty((B, 56), dtype=batch.dtype)

        for i, (name, extract_fn) in enumerate(var_dict.items()):
            features[:, i] = extract_fn(batch)
            if self.stats and name in self.stats:
                mu = self.stats[name]['mean']
                std = np.sqrt(self.stats[name]['variance']) + 1e-6
                features[:, i] = (features[:, i] - mu) / std
                if np.any(np.isnan(features[:, i])): 
                    print("warning! ", mu, std, " caused some nans")

        # Physics-aware normalization
        if self.norm:  # norm=True case
            # MET pt and phi
            features[:, 0] = features[:, 0] / 1200  # MET pt
            features[:, 1] = (features[:, 1] + np.pi) / (2 * np.pi)  # MET phi
            
            # Electrons: pt, eta, phi
            features[:, 2:6] = features[:, 2:6] / 1200  # Electron pt
            features[:, 6:10] = (features[:, 6:10] + 5) / 10  # Electron eta
            features[:, 10:14] = (features[:, 10:14] + np.pi) / (2 * np.pi)  # Electron phi
            
            # Muons: pt, eta, phi
            features[:, 14:18] = features[:, 14:18] / 800  # Muon pt
            features[:, 18:22] = (features[:, 18:22] + 5) / 10  # Muon eta
            features[:, 22:26] = (features[:, 22:26] + np.pi) / (2 * np.pi)  # Muon phi
            
            # Jets: pt, eta, phi
            features[:, 26:36] = features[:, 26:36] / 2500  # Jet pt
            features[:, 36:46] = (features[:, 36:46] + 5) / 10  # Jet eta
            features[:, 46:56] = (features[:, 46:56] + np.pi) / (2 * np.pi)  # Jet phi

        return features

        
        # Assign features directly into the pre-allocated array
        #features[:,  0]  = batch[:, 0, 0]   # MET pt
        #features[:,  1]  = batch[:, 0, 2]   # MET phi
    
        #features[:,  2:6]   = batch[:, 1:5, 0]  # Electron pt
        #features[:,  6:10]  = batch[:, 1:5, 1]  # Electron eta
        #features[:, 10:14]  = batch[:, 1:5, 2]  # Electron phi
    
        #features[:, 14:18]  = batch[:, 5:9, 0]  # Muon pt
        #features[:, 18:22]  = batch[:, 5:9, 1]  # Muon eta
        #features[:, 22:26]  = batch[:, 5:9, 2]  # Muon phi
    
        #features[:, 26:36]  = batch[:, 9:19, 0]  # Jet pt
        #features[:, 36:46]  = batch[:, 9:19, 1]  # Jet eta
        #features[:, 46:56]  = batch[:, 9:19, 2]  # Jet phi
    
        return features  # shape: (B, 56)

    def close(self):
        self.file.close()

    

def load_data(file_path, batch_size=64, shuffle=True, num_workers=0, drop_last=False, norm=False):
    dataset = HDF5Dataset(file_path, norm=norm)
    loader = DataLoader(dataset, 
                        batch_size=None, 
                        shuffle=False,  
                        sampler=RandomBatchSampler(dataset, batch_size, shuffle, drop_last),
                        num_workers=num_workers, 
                        collate_fn=None,
                        pin_memory=True # for faster gpu transfer
                       )

    return loader
