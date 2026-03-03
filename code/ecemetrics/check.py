import torch 
import pandas as pd
import numpy as np
import random
from ecemetrics.utils import seed_everything
import inspect


def check_boolean(weighted):
    """Ensure weighted is a Boolean"""
    if not isinstance(weighted, bool):
        raise ValueError("Input must be a Boolean")
    return True
    
def check_preds_tab_ok(preds, y): ### MODIFIED
    """
    Ensure pred is an array-like of the same type and length as y,
      with all values in [0, 1].
    """
    # Convert alpha and cover to arrays
    try:
        preds_arr = np.asarray(preds, dtype=float)
    except Exception:
        raise TypeError("preds must be an array.")

    y_arr = np.asarray(y)

    # Vérifie que alpha et cover ont le même type d'objet
    if type(preds) != type(y):
        raise TypeError("preds and y must be of the same type.")

    if preds_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("alpha must have the same length as cover.")

    if np.any(preds_arr < 0) or np.any(preds_arr > 1):
        raise ValueError("All values in preds must be between 0 and 1 (inclusive).")

    return preds_arr



def check_tabular_1D(x):
    """
    Check that x is a valid 1D tabular array/vector.
    - Must be 1D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    # Check dimensionality
    if x.ndim != 1:
        raise ValueError(f"x must be 1D (tabular vector), got shape {x.shape}")

    # NumPy array
    if isinstance(x, np.ndarray):
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"x must be np.ndarray or torch.Tensor, got {type(x)}")

def check_n_splits(n_splits):
    """
    Checks if n_splits is a valid integer greater than 0.
    
    Raises:
        TypeError: If n_splits is not an integer.
        ValueError: If n_splits is less than 1.
    """
    if not isinstance(n_splits, int):
        raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")
    return True

def check_tabular(X):
    """
    Check that X is a valid tabular array/matrix.
    - Must be 2D
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"X must be np.ndarray or torch.Tensor or dataframe, got {type(X)}"
    )

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (tabular), got shape {X.shape}")
    

def check_tabular_strict(X):
    """
    Check that X is a valid tabular array/matrix.
    - Must be 2D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (tabular), got shape {X.shape}")
    # NumPy array
    if isinstance(X, np.ndarray):
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(X, torch.Tensor):
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

def check_y(y): ### MODIFIED
    """Ensure y is a valid 1D vector of class labels (multi-class)."""
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.squeeze()

    if isinstance(y, torch.Tensor):
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}")
        if y.numel() == 0:
            raise ValueError("y must not be empty")
        # sklearn accepte int ou str, mais torch → typiquement int
        if not torch.is_floating_point(y) and not torch.is_integral(y):
            raise ValueError("y must contain numeric class labels")

    elif isinstance(y, (list, np.ndarray)):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if y.size == 0:
            raise ValueError("y must not be empty")
        # labels discrets attendus
        if y.dtype.kind not in {"i", "u", "f", "U", "S", "O"}:
            raise ValueError("y must contain discrete class labels")

    else:
        raise TypeError(
            "y must be a numpy array, list, pandas Series/DataFrame, or torch tensor"
        )

    return y


def check_emptyness(array):
    """
    Check if the input array or tensor is empty.
    
    Parameters:
        array: Can be a list, numpy array, or torch tensor.
    
    Raises:
        ValueError: If the input array is empty.
    """
    # For PyTorch tensors
    if isinstance(array, torch.Tensor):
        if array.numel() == 0:
            raise ValueError("The tensor is empty.")
    
    # For NumPy arrays
    elif isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The numpy array is empty.")
            
    else:
        raise TypeError("Input must be a list, tuple, numpy array, or torch tensor.")

    return True  # Optional: Return True if not empty



def check_consistency(x, y): ### MODIFIED
    """Ensure x and y have the same length and type."""
    # Convert pandas Series to NumPy arrays
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values

    # Check type
    if type(y) is not type(x):
        raise TypeError(f"cover and groups must be of the same type, got {type(y)} and {type(x)}")
    
    # Check length
    if len(y) != len(x):
        raise ValueError(
            f"cover and groups must have the same length, got cover of shape {getattr(y, 'shape', len(y))} "
            f"and groups of shape {getattr(x, 'shape', len(x))}"
        )
