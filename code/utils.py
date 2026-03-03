import random
import numpy as np
import torch


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def process_binary_probs(probs: np.ndarray) -> np.ndarray:
    probs = probs.astype(np.float32)
    if probs.ndim == 1: # 1D array of probabilities
        return probs
    elif probs.ndim == 2 and probs.shape[1] == 2: # 2D array of probability pairs [(p0, p1), ...]
        return probs[:, 1]
    else:
        raise ValueError(f"Invalid input shape: {probs.shape}."
                        f"1D array or 2D array with shape (n, 2)")

def binary_probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts binary probabilities to logits using the logit function (inverse sigmoid).
    Clips logits to the log of the tinyest normal float32 to avoid infinite logit values.
    """
    probs = process_binary_probs(probs)

    with np.errstate(divide="ignore"):
        log_p = np.log(probs)
        log_1_minus_p = np.log1p(-probs)

    thresh = np.log(np.finfo(np.float32).tiny)
    log_p = np.clip(log_p, a_min=thresh, a_max=-thresh).reshape(-1, 1)
    log_1_minus_p = np.clip(log_1_minus_p, a_min=thresh, a_max=-thresh).reshape(-1, 1)

    return log_p, log_1_minus_p

def invert_sigmoid(probs):
    """
    Compute the inverse sigmoid (logit) of probabilities.
    Maps input in the range (0, 1) back to real-valued logits.
    """
    # We use np.clip to avoid log(0) errors if the input is exactly 0 or 1
    # though mathematically, the inverse is undefined at the boundaries.
    return np.log(probs / (1 - probs))

def multiclass_probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Converts multiclass probabilities to logits using the log function.
    Clips logits to the log of the tinyest normal float32 to avoid infinite logit values.
    """
    probs = probs.astype(np.float64)
    thresh = np.log(np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore"):
        logits = np.log(probs)
    logits = np.clip(logits, a_min=thresh, a_max=None)
    return logits

def softmax(logits):
    """Helper to convert logits to probabilities safely."""
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(logits):
    """
    Compute the sigmoid of logits.
    Maps input to probabilities in the range [0, 1].
    """
    return 1 / (1 + np.exp(-logits))

def adaptive_top_class_ece(p_test: np.ndarray, y_test: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculates the Adaptive Top-Class (Top-1) Expected Calibration Error (ECE)
    using equal-mass bins (same number of samples per bin).
    
    Args:
        p_test: Array of shape (n_samples, n_classes) with predicted probabilities.
        y_test: Array of shape (n_samples,) with integer true class labels.
        n_bins: Number of bins to divide the sorted samples into.
        
    Returns:
        float: The Adaptive Expected Calibration Error.
    """
    confidences = np.max(p_test, axis=1)
    predictions = np.argmax(p_test, axis=1)
    accuracies = (predictions == y_test).astype(float)
    
    sort_indices = np.argsort(confidences)
    sorted_confidences = confidences[sort_indices]
    sorted_accuracies = accuracies[sort_indices]
    
    conf_bins = np.array_split(sorted_confidences, n_bins)
    acc_bins = np.array_split(sorted_accuracies, n_bins)
    
    ece = 0.0
    total_samples = len(confidences)
    
    for conf_bin, acc_bin in zip(conf_bins, acc_bins):
        bin_size = len(conf_bin)
        
        if bin_size == 0:
            continue
            
        prop_in_bin = bin_size / total_samples
        
        accuracy_in_bin = np.mean(acc_bin)
        avg_confidence_in_bin = np.mean(conf_bin)
        
        ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        
    return float(ece)