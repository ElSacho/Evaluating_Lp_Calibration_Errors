import torch
import numpy as np
import pandas as pd

from ecemetrics.check import *

def clip_for_under(p, f_x):
    """
    Adapts inputs to the mathematical formula:
    1_{f>0.5} * max(p,f) + 1_{f<0.5} * min(p,f) + 1_{f=0.5} * 0.5
    """
    g_f_x = np.asarray(p)
    f_x = np.asarray(f_x)

    result = np.copy(p)
    
    mask1 = f_x > 0.5
    result[mask1] = np.maximum(p[mask1], f_x[mask1])
    
    mask2 = (f_x < 0.5) 
    result[mask2] = np.minimum(p[mask2], f_x[mask2])
    
    mask3 = f_x == 0.5
    result[mask3] = 0.5
    
    return result

def clip_for_over(p, f_x):
    """
    Adapts inputs to the mathematical formula:
    1_{f>0.5} * min(p,f) + 1_{f<0.5} * max(p,f) + 1_{f=0.5} * 0.5
    """
    g_f_x = np.asarray(p)
    f_x = np.asarray(f_x)

    result = np.copy(p)
    
    mask1 = f_x > 0.5
    result[mask1] = np.minimum(p[mask1], f_x[mask1])
    
    mask2 = (f_x < 0.5) 
    result[mask2] = np.maximum(p[mask2], f_x[mask2])
    
    mask3 = f_x == 0.5
    result[mask3] = 0.5
    
    return result

def clip_over(x, val_min):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=val_min)

    elif isinstance(x, np.ndarray):
        return np.clip(x, a_min=val_min, a_max=None)

    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.clip(lower=val_min)

    elif isinstance(x, (int, float)):
        return max(x, val_min)

    else:
        raise TypeError(f"Input must be a torch.Tensor, np.ndarray, pd.Series, pd.DataFrame, or a number, got {type(x)}")

def generalized_norm_score(p, y, f_x, z=2):
    """
    Calculates the proper loss l_{f(X)}(p, Y) for a given z-norm.
    
    Parameters:
    p (np.array): The predicted probability/vector.
    y (np.array): The actual outcome vector.
    f_x (np.array): f(X), the reference point.
    z (int/float): The norm order (e.g., 2 for Euclidean).
    """
    p = np.asarray(p)
    y = np.asarray(y)
    f_x = np.asarray(f_x)
    
    # Calculate difference vector
    diff = p - f_x

    n, d = p.shape

    if d == 1:
        return L1_ECE(p, y, f_x)

    y_onehot = np.zeros((n, d))
    y_onehot[np.arange(n), y] = 1
    
    norm_val = np.linalg.norm(diff, ord=z, axis=1, keepdims=True)
    h_p = -norm_val.flatten() 
    
    # Handle subgradient safely (avoid division by zero)
    # We use np.where to handle the condition where p == f(X)
    safe_norm = np.where(norm_val == 0, 1.0, norm_val)
    
    # Formula for gradient of z-norm: (|u|^(z-1) * sgn(u)) / (||u||_z^(z-1))
    grad_norm = (np.abs(diff)**(z-1) * np.sign(diff)) / (safe_norm**(z-1))
    
    # Set subgradient to 0 where p == f_x
    subgradient = -np.where(norm_val == 0, 0, grad_norm)
    
    # Row-wise dot product: sum(subgradient * (y - p))
    dot_product = np.sum(subgradient * (y_onehot - p), axis=1)
    
    return dot_product + h_p

def norm_2_score(p, y, f_x):
    """Specific implementation for z=2 (Euclidean norm)"""
    return generalized_norm_score(p, y, f_x, z=2)

def make_generalized_norm_score(z):
    def loss(p, y, f_x):
        return generalized_norm_score(p, y, f_x, z=z)
    
    loss.__name__ = f"generalized_norm_score_{z}"
    return loss

def brier_score(pred_proba, y):
    pred_proba = np.asarray(pred_proba)
    y = np.asarray(y)

    check_y(y)

    n = y.shape[0]

    # Binary
    if pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
        p = pred_proba.squeeze()
        return (p - y) ** 2

    # Multi-class
    if pred_proba.ndim == 2:
        n_pred, d = pred_proba.shape
        if n_pred != n:
            raise ValueError("pred_proba and y must have the same number of samples")

        # one-hot 
        y_onehot = np.zeros((n, d))
        y_onehot[np.arange(n), y] = 1

        return np.sum((pred_proba - y_onehot) ** 2, axis=1)

    raise ValueError(f"Invalid pred_proba shape {pred_proba.shape}")

def logloss(pred_proba, y, eps=1e-8):
    # ---------- TORCH ----------
    if isinstance(y, torch.Tensor):
        if not isinstance(pred_proba, torch.Tensor):
            pred_proba = torch.tensor(pred_proba, dtype=torch.float32)

        pred_proba = torch.clamp(pred_proba, eps, 1 - eps)

        # binaire: (n, 1)
        if pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
            y = y.view(-1, 1)
            return -(
                y * torch.log(pred_proba) +
                (1 - y) * torch.log(1 - pred_proba)
            ).squeeze(1)

        # multi-classe: (n, d)
        if pred_proba.ndim == 2:
            idx = torch.arange(y.shape[0])
            return -torch.log(pred_proba[idx, y])

        raise ValueError(f"Invalid pred_proba shape {pred_proba.shape}")

    # ---------- NUMPY ----------
    else:
        pred_proba = np.asarray(pred_proba)
        y = np.asarray(y)

        pred_proba = np.clip(pred_proba, eps, 1 - eps)

        # binary: (n, 1)
        if pred_proba.ndim == 2 and pred_proba.shape[1] == 1:
            y = y.reshape(-1, 1)
            return -(
                y * np.log(pred_proba) +
                (1 - y) * np.log(1 - pred_proba)
            ).squeeze(1)

        # multi-class: (n, d)
        if pred_proba.ndim == 2:
            return -np.log(pred_proba[np.arange(y.shape[0]), y])

        raise ValueError(f"Invalid pred_proba shape {pred_proba.shape}")

def L1_ECE(g_f_x, y, f_x):
    if isinstance(g_f_x, (pd.Series, pd.DataFrame)):
        g_f_x = np.asarray(g_f_x)

    if isinstance(f_x, (pd.Series, pd.DataFrame)):
        f_x = np.asarray(f_x)

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y)

    g_f_x = np.asarray(g_f_x)
    f_x = np.asarray(f_x)
    
    y = np.asarray(y)

    if g_f_x.shape[1] == 1:
        out = y * 0.0 

        g_f_x = g_f_x.ravel()
        f_x = f_x.ravel()
        
        pos = g_f_x < f_x 
        neg = g_f_x > f_x 

        out[pos] = (f_x - y)[pos] 
        out[neg] = -(f_x - y)[neg] 

        return - out

    n, d = g_f_x.shape

    y_onehot = np.zeros((n, d))
    y_onehot[np.arange(n), y] = 1

    out = np.zeros_like(g_f_x)

    pos = g_f_x < f_x
    neg = g_f_x > f_x

    out[pos] = (f_x - y_onehot)[pos]
    out[neg] = -(f_x - y_onehot)[neg]

    # print("f_x", f_x[0])
    # print("pred_rec", pred_rectified[0])
    # print("y", y_onehot[0])
    # print("out", out[0])
    # print("sum",  -np.sum(out, axis=1)[0])
    # print('')

    return -np.sum(out, axis=1)

def brier_score_over(pred_proba, cover, alpha):
    return brier_score( clip_over(pred_proba, 1-alpha), cover)

def L1_ECE_over(g_f_x, y, f_x):
    return L1_ECE( clip_for_over(g_f_x, f_x), y, f_x)

def L1_ECE_under(g_f_x, y, f_x):
    return L1_ECE( clip_for_under(g_f_x, f_x), y, f_x)

