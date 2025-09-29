import jax
import jax.numpy as jnp

def evaluate_validation_error(val_loader, autoencoder, compute_error_batch):
    """
    Evaluate error on the entire validation set.
    
    Parameters
    ----------
    val_loader : DataLoader
        Validation data loader
    autoencoder : Model
        The autoencoder model
    compute_error_batch : function
        Function to compute error for a batch
        
    Returns
    -------
    float
        Average validation error
    """
    total_error = 0.0
    n_batches = 0
    
    # Evaluate in batches for efficiency
    for batch in val_loader:
        batch = jax.numpy.array(batch, dtype=jnp.float64)
        error = float(compute_error_batch(batch, *autoencoder.arrays))
        total_error += error
        n_batches += 1
    
    # Return average error across batches
    return total_error / n_batches if n_batches > 0 else 0.0


class SimpleEarlyStopping:
    """
    Simple early stopping to prevent overfitting.
    
    Parameters
    ----------
    patience : int
        Number of epochs with no improvement to wait before stopping
    min_delta : float
        Minimum change to consider as improvement
    """
    
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_model_state = None
        
    def check(self, val_loss, model, epoch):
        """
        Check if we should stop training.
        
        Parameters
        ----------
        val_loss : float
            Current validation loss
        model : Model
            Current model to potentially save
        epoch : int
            Current epoch number
            
        Returns
        -------
        bool
            True if training should stop
        """
        # Check if we have improvement
        if val_loss < (self.best_loss - self.min_delta):
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            
            # Save model state (deep copy of arrays)
            self.best_model_state = [arr.copy() for arr in model.arrays]
            return False
        else:
            # No improvement
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f"\n[EARLY STOPPING] No improvement for {self.patience} epochs.")
                print(f"Best validation error: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")
                return True
                
        return False
    
    def restore_best_model(self, model):
        """
        Restore the best model state.
        
        Parameters
        ----------
        model : Model
            Model to restore weights to
        """
        if self.best_model_state is not None:
            model.update_tensors(self.best_model_state)
            print(f"[RESTORED] Best model from epoch {self.best_epoch + 1}")

