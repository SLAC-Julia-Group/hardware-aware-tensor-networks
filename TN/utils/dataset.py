import h5py
import matplotlib.pyplot as plt
import math
import numpy as np


var_dict = {
    "met_pt":       lambda data: data[:, 0, 0],
    "met_phi":      lambda data: data[:, 0, 2],
    
    "electron0_pt": lambda data: data[:, 1, 0],
    "electron1_pt": lambda data: data[:, 2, 0],
    "electron2_pt": lambda data: data[:, 3, 0],
    "electron3_pt": lambda data: data[:, 4, 0],

    "electron0_eta": lambda data: data[:, 1, 1],
    "electron1_eta": lambda data: data[:, 2, 1],
    "electron2_eta": lambda data: data[:, 3, 1],
    "electron3_eta": lambda data: data[:, 4, 1],

    "electron0_phi": lambda data: data[:, 1, 2],
    "electron1_phi": lambda data: data[:, 2, 2],
    "electron2_phi": lambda data: data[:, 3, 2],
    "electron3_phi": lambda data: data[:, 4, 2],

    "mu0_pt": lambda data: data[:, 5, 0],
    "mu1_pt": lambda data: data[:, 6, 0],
    "mu2_pt": lambda data: data[:, 7, 0],
    "mu3_pt": lambda data: data[:, 8, 0],

    "mu0_eta": lambda data: data[:, 5, 1],
    "mu1_eta": lambda data: data[:, 6, 1],
    "mu2_eta": lambda data: data[:, 7, 1],
    "mu3_eta": lambda data: data[:, 8, 1],

    "mu0_phi": lambda data: data[:, 5, 2],
    "mu1_phi": lambda data: data[:, 6, 2],
    "mu2_phi": lambda data: data[:, 7, 2],
    "mu3_phi": lambda data: data[:, 8, 2],

    "jet0_pt":  lambda data: data[:, 9, 0],
    "jet1_pt":  lambda data: data[:, 10, 0],
    "jet2_pt":  lambda data: data[:, 11, 0],
    "jet3_pt":  lambda data: data[:, 12, 0],
    "jet4_pt":  lambda data: data[:, 13, 0],
    "jet5_pt":  lambda data: data[:, 14, 0],
    "jet6_pt":  lambda data: data[:, 15, 0],
    "jet7_pt":  lambda data: data[:, 16, 0],
    "jet8_pt":  lambda data: data[:, 17, 0],
    "jet9_pt":  lambda data: data[:, 18, 0],

    "jet0_eta": lambda data: data[:, 9, 1],
    "jet1_eta": lambda data: data[:, 10, 1],
    "jet2_eta": lambda data: data[:, 11, 1],
    "jet3_eta": lambda data: data[:, 12, 1],
    "jet4_eta":  lambda data: data[:, 13, 1],
    "jet5_eta":  lambda data: data[:, 14, 1],
    "jet6_eta":  lambda data: data[:, 15, 1],
    "jet7_eta":  lambda data: data[:, 16, 1],
    "jet8_eta":  lambda data: data[:, 17, 1],
    "jet9_eta":  lambda data: data[:, 18, 1],

    "jet0_phi": lambda data: data[:, 9, 2],
    "jet1_phi": lambda data: data[:, 10, 2],
    "jet2_phi": lambda data: data[:, 11, 2],
    "jet3_phi": lambda data: data[:, 12, 2],
    "jet4_phi":  lambda data: data[:, 13, 2],
    "jet5_phi":  lambda data: data[:, 14, 2],
    "jet6_phi":  lambda data: data[:, 15, 2],
    "jet7_phi":  lambda data: data[:, 16, 2],
    "jet8_phi":  lambda data: data[:, 17, 2],
    "jet9_phi":  lambda data: data[:, 18, 2],
}



def get_norm_dict(sample_path):
    # returns a dictionary with mean and variance for each variable defined in var_dict
    try:        
        stats = load_norm("norm_parameters.npz")
    except:

        f = h5py.File(sample_path + "/background_for_training.h5", "r")
        data = f['Particles']
        stats = get_distribution_stats(data, var_dict, n_samples=None) # all samples
        save_norm(stats, "norm_parameters.npz")

    return stats


def plot_all_variables(data, var_dict, n_samples=None, bins=50, figsize=(20, 20)):
    """
    Plot histograms for all variables in var_dict from the given data.

    Parameters
    ----------
    data : np.ndarray
   
    var_dict : dict
        Dictionary of variable names to lambda indexers (e.g., lambda d: d[:, 1, 0])
    n_samples : int or None
        Number of samples to use. If None, uses all data.
    bins : int
        Number of bins for histograms
    figsize : tuple
        Overall figure size
    """

    if n_samples is None:
        n_samples = data.shape[0]

    n_vars = len(var_dict)
    ncols = 4
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (name, extractor) in enumerate(var_dict.items()):
        try:
            values = extractor(data[:n_samples]).flatten()
            axes[i].hist(values, bins=bins, alpha=0.7, color='steelblue')
            axes[i].set_title(name)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center')
            axes[i].set_title(name)
    
    for ax in axes[n_vars:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_distribution_stats(data, var_dict, n_samples=None):
    """
    Compute mean and variance for each variable in var_dict.

    Parameters
    ----------
    data : np.ndarray
        Full dataset of shape (B, 19, 4)
    var_dict : dict
        Dictionary of variable names -> lambda extractors
    n_samples : int or None
        Number of samples to use. If None, uses all data.

    Returns
    -------
    stats_dict : dict
        Dictionary of variable -> {'mean': ..., 'variance': ...}
    """
    if n_samples is None:
        n_samples = data.shape[0]

    stats_dict = {}

    for name, extractor in var_dict.items():
        try:
            values = extractor(data[:n_samples]).flatten()
            stats_dict[name] = {
                'mean': float(np.mean(values)),
                'variance': float(np.var(values))
            }
        except Exception as e:
            stats_dict[name] = {
                'mean': None,
                'variance': None,
                'error': str(e)
            }

    return stats_dict


def save_norm(stats_dict, filename):
    flat_dict = {}
    for key, stats in stats_dict.items():
        flat_dict[f"{key}_mean"] = stats['mean']
        flat_dict[f"{key}_var"] = stats['variance']
    np.savez(filename, **flat_dict)

def load_norm(filename):
    data = np.load(filename)
    stats_dict = {}
    for key in data.files:
        base, typ = key.rsplit('_', 1)
        if base not in stats_dict:
            stats_dict[base] = {}
        stats_dict[base][typ if typ != 'var' else 'variance'] = float(data[key])
    return stats_dict
