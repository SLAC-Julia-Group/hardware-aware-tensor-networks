#!/usr/bin/env python3
"""
Export raw (unnormalized) HDF5 data to .dat files for HLS testbench.

This script:
1. Reads HDF5 files directly (background and signal)
2. Extracts 57 raw features per event (including MET eta = 0)
3. Saves to space-separated .dat files
4. Also runs JAX model inference to generate reference outputs
"""

import h5py
import numpy as np

def extract_raw_features(h5_data, event_indices, include_met_eta=True):
    """
    Extract raw features from HDF5 dataset.
    
    Parameters
    ----------
    h5_data : h5py.Dataset
        HDF5 dataset with shape (n_events, 19, 4) where:
        - 19 particles: [MET, 4e, 4mu, 10jets]
        - 4 features: [pt, eta, phi, mass] (we only use first 3)
    event_indices : list of int
        Which events to extract
    include_met_eta : bool
        If True, include MET eta=0 as a dummy feature (for 57 total)
        
    Returns
    -------
    features : np.ndarray
        Shape (n_events, 57) with RAW unnormalized features in order:
        [met_pt, met_eta, met_phi,
         e0_pt, e0_eta, e0_phi, e1_pt, e1_eta, e1_phi, ...,
         mu0_pt, mu0_eta, mu0_phi, mu1_pt, mu1_eta, mu1_phi, ...,
         jet0_pt, jet0_eta, jet0_phi, jet1_pt, jet1_eta, jet1_phi, ...]
    """
    n_events = len(event_indices)
    n_features = 57 if include_met_eta else 56
    
    features = np.zeros((n_events, n_features), dtype=np.float32)
    
    # Load the events
    batch = h5_data[event_indices]  # Shape: (n_events, 19, 4)
    
    idx = 0
    
    # MET (particle 0)
    features[:, idx] = batch[:, 0, 0]  # pt
    idx += 1
    if include_met_eta:
        features[:, idx] = 0.0  # eta (dummy, always 0 for MET)
        idx += 1
    features[:, idx] = batch[:, 0, 2]  # phi
    idx += 1
    
    # Electrons (particles 1-4)
    for i in range(1, 5):
        features[:, idx] = batch[:, i, 0]  # pt
        features[:, idx+1] = batch[:, i, 1]  # eta
        features[:, idx+2] = batch[:, i, 2]  # phi
        idx += 3
    
    # Muons (particles 5-8)
    for i in range(5, 9):
        features[:, idx] = batch[:, i, 0]  # pt
        features[:, idx+1] = batch[:, i, 1]  # eta
        features[:, idx+2] = batch[:, i, 2]  # phi
        idx += 3
    
    # Jets (particles 9-18)
    for i in range(9, 19):
        features[:, idx] = batch[:, i, 0]  # pt
        features[:, idx+1] = batch[:, i, 1]  # eta
        features[:, idx+2] = batch[:, i, 2]  # phi
        idx += 3
    
    assert idx == n_features, f"Feature extraction error: expected {n_features}, got {idx}"
    
    return features


def write_dat_file(features, filename):
    """
    Write features to space-separated .dat file.
    
    Parameters
    ----------
    features : np.ndarray
        Shape (n_events, n_features)
    filename : str
        Output file path
    """
    with open(filename, 'w') as f:
        for event in features:
            # Write all 57 features space-separated on one line
            line = ' '.join(f'{val:.6f}' for val in event)
            f.write(line + '\n')
    
    print(f"Wrote {len(features)} events to {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEvents', type=int, default=5, help='Number of events to extract')
    parser.add_argument('--inFile', type=str, 
                       default="/lus/eagle/projects/ATLAS_workflow_ALCF/sagar/QiML/background_for_training.h5")
    parser.add_argument('--outFile', type=str,
                       default="background.dat")
    args = parser.parse_args()
        
    print(f"Extracting {args.nEvents} events...")
    with h5py.File(args.inFile, 'r') as f:
        data = f['Particles']
        indices = list(range(args.nEvents))
        features = extract_raw_features(data, indices, include_met_eta=True)

    write_dat_file(features, args.outFile)

    print(f"\nDone! Created:")
    print(f"  {args.outFile}")

if __name__ == "__main__":
    main()