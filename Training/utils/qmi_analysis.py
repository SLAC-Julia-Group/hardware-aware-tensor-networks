"""
Quantum Mutual Information (QMI) Analysis for Tensor Network Site Ordering.

This module computes the pairwise QMI matrix from embedded training data to determine
optimal site ordering for MPS-based models. By placing highly correlated particles
adjacent in the MPS chain, we can achieve the same expressivity with smaller bond
dimensions.

Theory
------
For product state inputs |ψ^(n)⟩ = ⊗_i |û_i^(n)⟩, the mixed state over the training
distribution is:

    ρ = (1/N) Σ_n |ψ^(n)⟩⟨ψ^(n)|

The QMI between sites i and j is:

    I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij})

where S(ρ) = -Tr(ρ log ρ) is the von Neumann entropy.

For product states, the reduced density matrices are:
    ρ_i = (1/N) Σ_n |û_i^(n)⟩⟨û_i^(n)|
    ρ_{ij} = (1/N) Σ_n |û_i^(n)⟩⟨û_i^(n)| ⊗ |û_j^(n)⟩⟨û_j^(n)|

Note: ρ_{ij} ≠ ρ_i ⊗ ρ_j in general, which is why QMI can be non-zero even for
product states - it captures classical correlations in the data distribution.

Usage
-----
    from utils.qmi_analysis import compute_qmi_matrix, spectral_ordering, visualize_qmi
    from utils.dataloader import create_split_dataloaders
    
    # Load training data
    train_loader, _, _ = create_split_dataloaders(
        "background_for_training.h5",
        batch_size=10000,
        norm=True,
        skip_MET_eta=False
    )
    
    # Compute QMI matrix
    qmi_matrix = compute_qmi_matrix(train_loader)
    
    # Get optimal ordering
    ordering = spectral_ordering(qmi_matrix)
    
    # Visualize
    visualize_qmi(qmi_matrix, save_path="qmi_heatmap.png")
    
    # Use ordering in training (pass to dataloader)
    train_loader, _, _ = create_split_dataloaders(
        "background_for_training.h5",
        particle_ordering=ordering,
        ...
    )

Author: QiML Project
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt


# Particle configuration for the 40MHz dataset
PARTICLE_CONFIG = {
    'n_particles': 19,
    'features_per_particle': 3,
    'particle_names': [
        'MET',
        'e0', 'e1', 'e2', 'e3',
        'μ0', 'μ1', 'μ2', 'μ3',
        'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9'
    ],
    'particle_types': ['MET'] + ['Electron']*4 + ['Muon']*4 + ['Jet']*10,
    'type_colors': {
        'MET': '#e41a1c',      # Red
        'Electron': '#377eb8', # Blue
        'Muon': '#4daf4a',     # Green
        'Jet': '#ff7f00'       # Orange
    }
}


# =============================================================================
# Ordering utilities (for use with dataloader)
# =============================================================================

def get_feature_ordering(particle_ordering: np.ndarray, 
                         skip_MET_eta: bool = False) -> np.ndarray:
    """
    Convert particle ordering to feature ordering for reordering raw features.
    
    Given a particle ordering that maps new_particle_idx -> original_particle_idx,
    this function computes the corresponding feature index mapping.
    
    Parameters
    ----------
    particle_ordering : np.ndarray
        Array of shape (n_particles,) mapping new index to original particle index.
        Example: [4, 3, 18, ...] means new site 0 = original particle 4 (e3).
    skip_MET_eta : bool
        If True, MET has only 2 features (pt, phi). If False, MET has 3 features
        like all other particles.
        
    Returns
    -------
    np.ndarray
        Feature ordering array that can be used as: reordered = features[:, feature_ordering]
        
    Example
    -------
    >>> ordering = np.array([4, 3, 0, 1, 2])  # Simple 5-particle example
    >>> feat_ord = get_feature_ordering(ordering[:5], skip_MET_eta=False)
    >>> # feat_ord reorders features so particle 4's features come first, then 3's, etc.
    """
    feature_ordering = []
    
    if skip_MET_eta:
        # MET has 2 features, others have 3
        # Original feature layout:
        #   MET: [0, 1] (pt, phi)
        #   e0:  [2, 3, 4]
        #   e1:  [5, 6, 7]
        #   ...
        def get_particle_features(orig_idx):
            if orig_idx == 0:  # MET
                return [0, 1]
            else:
                # Particle i (i > 0) starts at feature 2 + (i-1)*3
                start = 2 + (orig_idx - 1) * 3
                return [start, start + 1, start + 2]
    else:
        # All particles have 3 features
        def get_particle_features(orig_idx):
            start = orig_idx * 3
            return [start, start + 1, start + 2]
    
    for orig_idx in particle_ordering:
        feature_ordering.extend(get_particle_features(orig_idx))
    
    return np.array(feature_ordering, dtype=np.int64)


def get_reordered_particle_names(particle_ordering: np.ndarray,
                                  particle_names: Optional[List[str]] = None) -> List[str]:
    """
    Get particle names in the new ordering.
    
    Parameters
    ----------
    particle_ordering : np.ndarray
        Particle ordering array (new_idx -> original_idx).
    particle_names : list of str, optional
        Original particle names. Defaults to standard 40MHz naming.
        
    Returns
    -------
    list of str
        Particle names in the new order.
    """
    if particle_names is None:
        particle_names = PARTICLE_CONFIG['particle_names']
    
    return [particle_names[i] for i in particle_ordering]


def save_ordering_config(particle_ordering: np.ndarray,
                         output_path: str,
                         skip_MET_eta: bool = False) -> None:
    """
    Save ordering configuration as a JSON file for easy loading in training scripts.
    
    Parameters
    ----------
    particle_ordering : np.ndarray
        Particle ordering array.
    output_path : str
        Path to save the JSON config.
    skip_MET_eta : bool
        Whether MET eta is skipped.
    """
    import json
    
    feature_ordering = get_feature_ordering(particle_ordering, skip_MET_eta)
    reordered_names = get_reordered_particle_names(particle_ordering)
    
    config = {
        'particle_ordering': particle_ordering.tolist(),
        'feature_ordering': feature_ordering.tolist(),
        'reordered_particle_names': reordered_names,
        'skip_MET_eta': skip_MET_eta,
        'n_particles': len(particle_ordering),
        'n_features': len(feature_ordering)
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved ordering config to {output_path}")


def load_ordering_config(config_path: str) -> dict:
    """
    Load ordering configuration from JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to the JSON config file.
        
    Returns
    -------
    dict
        Configuration dictionary with 'particle_ordering', 'feature_ordering', etc.
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert lists back to numpy arrays
    config['particle_ordering'] = np.array(config['particle_ordering'])
    config['feature_ordering'] = np.array(config['feature_ordering'])
    
    return config


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Normalize vectors to unit length.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n_samples, d) containing vectors to normalize.
    eps : float
        Small constant to avoid division by zero.
        
    Returns
    -------
    np.ndarray
        Normalized vectors of shape (n_samples, d).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def compute_single_site_density_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute single-site reduced density matrix from normalized vectors.
    
    For normalized vectors |û^(n)⟩, computes:
        ρ = (1/N) Σ_n |û^(n)⟩⟨û^(n)|
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (n_samples, d) containing normalized vectors.
        
    Returns
    -------
    np.ndarray
        Density matrix of shape (d, d).
    """
    n_samples = vectors.shape[0]
    # Outer product sum: (N, d, 1) @ (N, 1, d) -> (N, d, d) -> sum -> (d, d)
    rho = np.einsum('ni,nj->ij', vectors, vectors) / n_samples
    return rho


def compute_two_site_density_matrix(vectors_i: np.ndarray, 
                                     vectors_j: np.ndarray) -> np.ndarray:
    """
    Compute two-site reduced density matrix from normalized vectors.
    
    For normalized vectors |û_i^(n)⟩ and |û_j^(n)⟩, computes:
        ρ_{ij} = (1/N) Σ_n |û_i^(n)⟩⟨û_i^(n)| ⊗ |û_j^(n)⟩⟨û_j^(n)|
    
    Parameters
    ----------
    vectors_i : np.ndarray
        Array of shape (n_samples, d_i) for site i.
    vectors_j : np.ndarray
        Array of shape (n_samples, d_j) for site j.
        
    Returns
    -------
    np.ndarray
        Density matrix of shape (d_i * d_j, d_i * d_j).
    """
    n_samples = vectors_i.shape[0]
    d_i = vectors_i.shape[1]
    d_j = vectors_j.shape[1]
    
    # For each sample, compute |û_i⟩⟨û_i| ⊗ |û_j⟩⟨û_j|
    # This is equivalent to |û_i ⊗ û_j⟩⟨û_i ⊗ û_j|
    # where |û_i ⊗ û_j⟩ is the tensor product (Kronecker product of vectors)
    
    # Compute tensor product vectors: (N, d_i * d_j)
    tensor_products = np.einsum('ni,nj->nij', vectors_i, vectors_j).reshape(n_samples, -1)
    
    # Compute density matrix
    rho = np.einsum('ni,nj->ij', tensor_products, tensor_products) / n_samples
    return rho


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).
    
    Parameters
    ----------
    rho : np.ndarray
        Density matrix (must be Hermitian, positive semi-definite).
    eps : float
        Small constant for numerical stability in log computation.
        
    Returns
    -------
    float
        Von Neumann entropy in nats (natural log).
    """
    # Compute eigenvalues (density matrix is Hermitian)
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Filter out numerical noise (small negative or very small positive)
    eigenvalues = eigenvalues[eigenvalues > eps]
    
    # S = -Σ λ log(λ)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return entropy


def compute_qmi(rho_i: np.ndarray, 
                rho_j: np.ndarray, 
                rho_ij: np.ndarray) -> float:
    """
    Compute quantum mutual information I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
    
    Parameters
    ----------
    rho_i : np.ndarray
        Single-site density matrix for site i.
    rho_j : np.ndarray
        Single-site density matrix for site j.
    rho_ij : np.ndarray
        Two-site density matrix for sites i and j.
        
    Returns
    -------
    float
        Quantum mutual information (non-negative).
    """
    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)
    
    qmi = S_i + S_j - S_ij
    
    # QMI should be non-negative; small negative values are numerical artifacts
    return max(0.0, qmi)


def compute_qmi_matrix(dataloader,
                       n_particles: int = 19,
                       features_per_particle: int = 3,
                       max_batches: Optional[int] = None,
                       verbose: bool = True) -> np.ndarray:
    """
    Compute the pairwise QMI matrix from training data.
    
    This function accumulates density matrices across all batches, then computes
    QMI for each pair of sites. Memory efficient for large datasets.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader yielding batches of shape (batch_size, n_features).
        Expected n_features = n_particles * features_per_particle.
    n_particles : int
        Number of particles (sites in the MPS).
    features_per_particle : int
        Number of features per particle (local Hilbert space dimension).
    max_batches : int, optional
        Maximum number of batches to process. None for all batches.
    verbose : bool
        Whether to show progress bar.
        
    Returns
    -------
    np.ndarray
        Symmetric QMI matrix of shape (n_particles, n_particles).
        Diagonal elements are zero (QMI of a site with itself is not defined
        in the classical correlation sense we care about).
    """
    d = features_per_particle
    n_features = n_particles * features_per_particle
    
    # Accumulators for density matrices
    # Single-site: (n_particles, d, d)
    rho_single = np.zeros((n_particles, d, d), dtype=np.float64)
    
    # Two-site: only upper triangle, (n_pairs, d*d, d*d)
    # We'll compute these lazily to save memory
    n_pairs = n_particles * (n_particles - 1) // 2
    rho_two = np.zeros((n_pairs, d*d, d*d), dtype=np.float64)
    
    # Create pair index mapping
    pair_to_idx = {}
    idx = 0
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            pair_to_idx[(i, j)] = idx
            idx += 1
    
    total_samples = 0
    
    # Process batches
    iterator = tqdm(dataloader, desc="Computing density matrices") if verbose else dataloader
    
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        # Convert to numpy if needed
        if hasattr(batch, 'numpy'):
            batch = batch.numpy()
        batch = np.asarray(batch, dtype=np.float64)
        
        batch_size = batch.shape[0]
        total_samples += batch_size
        
        # Reshape to (batch_size, n_particles, features_per_particle)
        if batch.shape[1] != n_features:
            raise ValueError(
                f"Expected {n_features} features, got {batch.shape[1]}. "
                f"Check n_particles={n_particles} and features_per_particle={features_per_particle}."
            )
        
        particles = batch.reshape(batch_size, n_particles, features_per_particle)
        
        # Normalize each particle vector
        for p in range(n_particles):
            particles[:, p, :] = normalize_vectors(particles[:, p, :])
        
        # Accumulate single-site density matrices
        for i in range(n_particles):
            vectors_i = particles[:, i, :]  # (batch_size, d)
            rho_single[i] += np.einsum('ni,nj->ij', vectors_i, vectors_i)
        
        # Accumulate two-site density matrices
        for i in range(n_particles):
            vectors_i = particles[:, i, :]
            for j in range(i+1, n_particles):
                vectors_j = particles[:, j, :]
                
                # Tensor product: (batch_size, d*d)
                tensor_prod = np.einsum('ni,nj->nij', vectors_i, vectors_j).reshape(batch_size, -1)
                
                pair_idx = pair_to_idx[(i, j)]
                rho_two[pair_idx] += np.einsum('ni,nj->ij', tensor_prod, tensor_prod)
    
    if verbose:
        print(f"Processed {total_samples:,} samples")
    
    # Normalize by total samples
    rho_single /= total_samples
    rho_two /= total_samples
    
    # Compute QMI matrix
    qmi_matrix = np.zeros((n_particles, n_particles), dtype=np.float64)
    
    if verbose:
        print("Computing QMI values...")
    
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            pair_idx = pair_to_idx[(i, j)]
            
            qmi_val = compute_qmi(rho_single[i], rho_single[j], rho_two[pair_idx])
            
            qmi_matrix[i, j] = qmi_val
            qmi_matrix[j, i] = qmi_val  # Symmetric
    
    return qmi_matrix


def spectral_ordering(qmi_matrix: np.ndarray, 
                      return_fiedler: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute optimal site ordering using spectral graph theory.
    
    Uses the Fiedler vector (second smallest eigenvector of the graph Laplacian)
    to determine an ordering that places highly connected nodes nearby.
    
    The QMI matrix is treated as a weighted adjacency matrix of a graph where
    edge weights represent the strength of correlation between sites.
    
    Parameters
    ----------
    qmi_matrix : np.ndarray
        Symmetric QMI matrix of shape (n, n).
    return_fiedler : bool
        If True, also return the Fiedler vector.
        
    Returns
    -------
    ordering : np.ndarray
        Permutation array mapping new_index -> original_index.
        To reorder data: reordered_data = original_data[:, ordering]
    fiedler_vector : np.ndarray, optional
        The Fiedler vector (returned if return_fiedler=True).
    """
    n = qmi_matrix.shape[0]
    
    # Construct graph Laplacian: L = D - A
    # where D is the degree matrix (diagonal) and A is the adjacency matrix
    degree = np.sum(qmi_matrix, axis=1)
    laplacian = np.diag(degree) - qmi_matrix
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    # Fiedler vector is the eigenvector corresponding to second smallest eigenvalue
    # (smallest is always 0 for connected graphs, with eigenvector = constant)
    fiedler_idx = 1
    fiedler_vector = eigenvectors[:, fiedler_idx]
    
    # Sort sites by Fiedler vector values
    ordering = np.argsort(fiedler_vector)
    
    if return_fiedler:
        return ordering, fiedler_vector
    return ordering


def hierarchical_ordering(qmi_matrix: np.ndarray,
                          method: str = 'average') -> np.ndarray:
    """
    Compute site ordering using hierarchical clustering.
    
    Alternative to spectral ordering that may work better for some correlation
    structures.
    
    Parameters
    ----------
    qmi_matrix : np.ndarray
        Symmetric QMI matrix of shape (n, n).
    method : str
        Linkage method for hierarchical clustering.
        Options: 'single', 'complete', 'average', 'ward'.
        
    Returns
    -------
    ordering : np.ndarray
        Permutation array mapping new_index -> original_index.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    # Convert QMI (similarity) to distance
    # Higher QMI = more similar = smaller distance
    max_qmi = np.max(qmi_matrix)
    if max_qmi > 0:
        distance_matrix = max_qmi - qmi_matrix
    else:
        distance_matrix = np.ones_like(qmi_matrix)
    
    # Set diagonal to 0 (distance to self)
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix)
    
    # Hierarchical clustering
    Z = linkage(condensed, method=method)
    
    # Get leaf ordering
    ordering = leaves_list(Z)
    
    return ordering


def visualize_qmi(qmi_matrix: np.ndarray,
                  particle_names: Optional[List[str]] = None,
                  title: str = "Quantum Mutual Information Matrix",
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (12, 10),
                  cmap: str = 'viridis',
                  show_values: bool = False,
                  ordering: Optional[np.ndarray] = None,
                  log_scale: bool = False,
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None) -> plt.Figure:
    """
    Visualize the QMI matrix as a heatmap.
    
    Parameters
    ----------
    qmi_matrix : np.ndarray
        QMI matrix of shape (n_particles, n_particles).
    particle_names : list of str, optional
        Names for each particle. Defaults to standard 40MHz naming.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure. If None, figure is displayed.
    figsize : tuple
        Figure size (width, height).
    cmap : str
        Colormap name.
    show_values : bool
        Whether to annotate cells with QMI values.
    ordering : np.ndarray, optional
        If provided, reorder the matrix according to this permutation.
    log_scale : bool
        If True, use logarithmic color scale. Useful when QMI values span
        multiple orders of magnitude.
    vmin : float, optional
        Minimum value for color scale. For log_scale, this sets the floor.
    vmax : float, optional
        Maximum value for color scale.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    from matplotlib.colors import LogNorm, Normalize
    
    if particle_names is None:
        particle_names = PARTICLE_CONFIG['particle_names']
    
    n = qmi_matrix.shape[0]
    
    # Apply ordering if provided
    if ordering is not None:
        qmi_display = qmi_matrix[np.ix_(ordering, ordering)]
        names_display = [particle_names[i] for i in ordering]
    else:
        qmi_display = qmi_matrix
        names_display = particle_names
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color normalization
    # For diagonal elements (self-QMI = 0), we need special handling in log scale
    qmi_off_diag = qmi_display.copy()
    np.fill_diagonal(qmi_off_diag, np.nan)  # Exclude diagonal for vmin/vmax calculation
    
    if vmax is None:
        vmax = np.nanmax(qmi_off_diag)
    if vmin is None:
        # For log scale, find minimum non-zero value
        positive_vals = qmi_off_diag[qmi_off_diag > 0]
        if len(positive_vals) > 0:
            vmin = np.nanmin(positive_vals) if log_scale else 0
        else:
            vmin = 1e-10 if log_scale else 0
    
    if log_scale:
        # Replace zeros and diagonal with vmin for visualization
        qmi_plot = qmi_display.copy()
        qmi_plot[qmi_plot <= 0] = vmin
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        qmi_plot = qmi_display
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create heatmap
    im = ax.imshow(qmi_plot, cmap=cmap, aspect='equal', norm=norm)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    scale_label = 'QMI (nats, log scale)' if log_scale else 'QMI (nats)'
    cbar.set_label(scale_label, fontsize=12)
    
    # Labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names_display, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names_display, fontsize=10)
    
    # Add value annotations if requested
    if show_values:
        for i in range(n):
            for j in range(n):
                if i != j:
                    text_color = 'white' if qmi_display[i, j] > qmi_display.max() / 2 else 'black'
                    ax.text(j, i, f'{qmi_display[i, j]:.3f}',
                           ha='center', va='center', color=text_color, fontsize=7)
    
    # Add grid lines to separate particle types
    # MET: 0, Electrons: 1-4, Muons: 5-8, Jets: 9-18
    if ordering is None:
        type_boundaries = [0.5, 4.5, 8.5]
        for b in type_boundaries:
            ax.axhline(y=b, color='white', linewidth=2, alpha=0.7)
            ax.axvline(x=b, color='white', linewidth=2, alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Particle', fontsize=12)
    ax.set_ylabel('Particle', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved QMI heatmap to {save_path}")
    
    return fig


def visualize_ordering_comparison(qmi_matrix: np.ndarray,
                                   ordering: np.ndarray,
                                   particle_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (20, 8),
                                   log_scale: bool = False) -> plt.Figure:
    """
    Side-by-side comparison of original and reordered QMI matrices.
    
    Parameters
    ----------
    qmi_matrix : np.ndarray
        Original QMI matrix.
    ordering : np.ndarray
        Optimal ordering permutation.
    particle_names : list of str, optional
        Names for each particle.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    log_scale : bool
        If True, use logarithmic color scale.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    from matplotlib.colors import LogNorm, Normalize
    
    if particle_names is None:
        particle_names = PARTICLE_CONFIG['particle_names']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute normalization parameters (shared between both plots)
    qmi_off_diag = qmi_matrix.copy()
    np.fill_diagonal(qmi_off_diag, np.nan)
    vmax = np.nanmax(qmi_off_diag)
    positive_vals = qmi_off_diag[qmi_off_diag > 0]
    vmin = np.nanmin(positive_vals) if (log_scale and len(positive_vals) > 0) else (1e-10 if log_scale else 0)
    
    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=0, vmax=vmax)
    
    # Prepare matrices for plotting
    def prep_matrix(mat):
        if log_scale:
            m = mat.copy()
            m[m <= 0] = vmin
            return m
        return mat
    
    # Original ordering
    ax = axes[0]
    im = ax.imshow(prep_matrix(qmi_matrix), cmap='viridis', aspect='equal', norm=norm)
    ax.set_xticks(np.arange(len(particle_names)))
    ax.set_yticks(np.arange(len(particle_names)))
    ax.set_xticklabels(particle_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(particle_names, fontsize=9)
    ax.set_title('Original Ordering', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add type boundaries
    for b in [0.5, 4.5, 8.5]:
        ax.axhline(y=b, color='white', linewidth=2, alpha=0.7)
        ax.axvline(x=b, color='white', linewidth=2, alpha=0.7)
    
    # Reordered
    ax = axes[1]
    qmi_reordered = qmi_matrix[np.ix_(ordering, ordering)]
    names_reordered = [particle_names[i] for i in ordering]
    
    im = ax.imshow(prep_matrix(qmi_reordered), cmap='viridis', aspect='equal', norm=norm)
    ax.set_xticks(np.arange(len(particle_names)))
    ax.set_yticks(np.arange(len(particle_names)))
    ax.set_xticklabels(names_reordered, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names_reordered, fontsize=9)
    ax.set_title('Spectral Ordering (Optimized)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return fig


def print_ordering_summary(qmi_matrix: np.ndarray,
                           ordering: np.ndarray,
                           particle_names: Optional[List[str]] = None) -> None:
    """
    Print a summary of the optimal ordering and its benefits.
    
    Parameters
    ----------
    qmi_matrix : np.ndarray
        QMI matrix.
    ordering : np.ndarray
        Optimal ordering permutation.
    particle_names : list of str, optional
        Names for each particle.
    """
    if particle_names is None:
        particle_names = PARTICLE_CONFIG['particle_names']
    
    n = len(ordering)
    
    print("\n" + "="*70)
    print("QMI ANALYSIS SUMMARY")
    print("="*70)
    
    # Optimal ordering
    print("\nOptimal particle ordering (left to right in MPS chain):")
    print("-" * 50)
    ordered_names = [particle_names[i] for i in ordering]
    for new_idx, orig_idx in enumerate(ordering):
        print(f"  Site {new_idx:2d}: {particle_names[orig_idx]:4s} (original index {orig_idx})")
    
    # Compute nearest-neighbor QMI sum (measure of locality)
    def nn_qmi_sum(matrix):
        """Sum of nearest-neighbor QMI values."""
        total = 0
        for i in range(matrix.shape[0] - 1):
            total += matrix[i, i+1]
        return total
    
    original_nn_qmi = nn_qmi_sum(qmi_matrix)
    reordered_matrix = qmi_matrix[np.ix_(ordering, ordering)]
    reordered_nn_qmi = nn_qmi_sum(reordered_matrix)
    
    print(f"\nNearest-neighbor QMI sum:")
    print(f"  Original ordering:  {original_nn_qmi:.4f}")
    print(f"  Optimized ordering: {reordered_nn_qmi:.4f}")
    print(f"  Improvement:        {(reordered_nn_qmi/original_nn_qmi - 1)*100:+.1f}%")
    
    # Top correlated pairs
    print("\nTop 10 most correlated particle pairs:")
    print("-" * 50)
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, qmi_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, j, qmi_val in pairs[:10]:
        # Check if they're adjacent in optimal ordering
        new_i = np.where(ordering == i)[0][0]
        new_j = np.where(ordering == j)[0][0]
        adjacent = abs(new_i - new_j) == 1
        adj_marker = " ← adjacent" if adjacent else ""
        
        print(f"  {particle_names[i]:4s} - {particle_names[j]:4s}: QMI = {qmi_val:.4f}{adj_marker}")
    
    # Particle type statistics
    print("\nAverage QMI by particle type pair:")
    print("-" * 50)
    
    types = ['MET', 'Electron', 'Muon', 'Jet']
    type_indices = {
        'MET': [0],
        'Electron': [1, 2, 3, 4],
        'Muon': [5, 6, 7, 8],
        'Jet': list(range(9, 19))
    }
    
    for t1 in types:
        for t2 in types:
            if types.index(t1) <= types.index(t2):
                qmi_vals = []
                for i in type_indices[t1]:
                    for j in type_indices[t2]:
                        if i != j:
                            qmi_vals.append(qmi_matrix[i, j])
                if qmi_vals:
                    avg_qmi = np.mean(qmi_vals)
                    print(f"  {t1:8s} - {t2:8s}: {avg_qmi:.4f}")
    
    print("\n" + "="*70)


def analyze_qmi(dataloader,
                output_dir: str = ".",
                max_batches: Optional[int] = None,
                log_scale: bool = True,
                skip_MET_eta: bool = False,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete QMI analysis pipeline: compute matrix, find ordering, visualize.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for training data.
    output_dir : str
        Directory to save output files.
    max_batches : int, optional
        Maximum batches to process.
    log_scale : bool
        Whether to use log scale for visualizations (recommended when QMI values
        span multiple orders of magnitude).
    skip_MET_eta : bool
        Whether MET eta was skipped in data loading. Needed for correct feature
        ordering in the saved config.
    verbose : bool
        Whether to print progress and summary.
        
    Returns
    -------
    qmi_matrix : np.ndarray
        The computed QMI matrix.
    ordering : np.ndarray
        The optimal site ordering.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute QMI matrix
    qmi_matrix = compute_qmi_matrix(dataloader, max_batches=max_batches, verbose=verbose)
    
    # Save raw matrix
    np.save(os.path.join(output_dir, "qmi_matrix.npy"), qmi_matrix)
    if verbose:
        print(f"Saved QMI matrix to {output_dir}/qmi_matrix.npy")
    
    # Compute optimal ordering
    ordering, fiedler = spectral_ordering(qmi_matrix, return_fiedler=True)
    
    # Save ordering (numpy)
    np.save(os.path.join(output_dir, "optimal_ordering.npy"), ordering)
    if verbose:
        print(f"Saved optimal ordering to {output_dir}/optimal_ordering.npy")
    
    # Save ordering config (JSON - includes feature ordering for dataloader)
    save_ordering_config(
        ordering, 
        os.path.join(output_dir, "ordering_config.json"),
        skip_MET_eta=skip_MET_eta
    )
    
    # Visualizations
    visualize_qmi(
        qmi_matrix,
        title="QMI Matrix (Original Ordering)",
        save_path=os.path.join(output_dir, "qmi_original.png"),
        log_scale=log_scale
    )
    plt.close()
    
    visualize_ordering_comparison(
        qmi_matrix,
        ordering,
        save_path=os.path.join(output_dir, "qmi_comparison.png"),
        log_scale=log_scale
    )
    plt.close()
    
    # Print summary
    if verbose:
        print_ordering_summary(qmi_matrix, ordering)
    
    return qmi_matrix, ordering


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Fix Python path for command-line execution
    # This script lives in TN/utils/, so we need TN/ in the path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tn_dir = os.path.dirname(script_dir)  # TN/
    
    # Add TN/ to path so "from utils.X import Y" works
    if tn_dir not in sys.path:
        sys.path.insert(0, tn_dir)
    
    parser = argparse.ArgumentParser(
        description="Compute QMI matrix and optimal site ordering for tensor network training."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HDF5 training data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qmi_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (None for all)"
    )
    parser.add_argument(
        "--skip_met_eta",
        action="store_true",
        help="Skip MET eta feature (use 56 features instead of 57)"
    )
    
    args = parser.parse_args()
    
    # Import dataloader (now that path is fixed)
    from utils.dataloader import create_split_dataloaders
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    train_loader, _, _ = create_split_dataloaders(
        file_path=args.data_path,
        batch_size=args.batch_size,
        split_ratios=(0.70, 0.05, 0.25),
        norm=True,
        skip_MET_eta=args.skip_met_eta,
        shuffle=False
    )
    
    # Determine number of particles
    n_particles = 18 if args.skip_met_eta else 19
    
    # Run analysis
    qmi_matrix, ordering = analyze_qmi(
        train_loader,
        output_dir=args.output_dir,
        max_batches=args.max_batches
    )
    
    print("\nDone!")