import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from utils import *

class MulticlassCalibrationSimulator:
    def __init__(self, h_func, n_classes, alpha=None):
        self.n_classes = n_classes
        if alpha is None:
            self.alpha = np.ones(self.n_classes)
        else: 
            self.alpha = alpha
        self.h_func = h_func

    def generate_preds(self, n_samples=1000):
        preds = np.random.dirichlet(self.alpha, n_samples)
        
        return preds

    def generate_labels(self, preds):
        true_probs = self.h_func(preds) 
        true_probs = true_probs / true_probs.sum(axis=1, keepdims=True)

        labels = np.array([np.random.multinomial(1, p) for p in true_probs])        
        label_indices = np.argmax(labels, axis=1)
        
        return label_indices
    
    def calculate_true_L1_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        
        true_prob_vectors = self.h_func(preds)
        true_prob_vectors = true_prob_vectors / true_prob_vectors.sum(axis=1, keepdims=True)
        
        vector_l1_errors = np.sum(np.abs(preds - true_prob_vectors), axis=1)
        true_vector_ece = np.mean(vector_l1_errors)
        
        return true_vector_ece
    
    def calculate_true_Lz_ece(self, z=2, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        
        true_prob_vectors = self.h_func(preds)
        true_prob_vectors = true_prob_vectors / true_prob_vectors.sum(axis=1, keepdims=True)
        
        diff = preds - true_prob_vectors
        vector_lz_errors = np.linalg.norm(diff, ord=z, axis=1, keepdims=True)
        true_vector_ece = np.mean(vector_lz_errors)
        
        return true_vector_ece
    
    def calculate_true_KL_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        
        true_prob_vectors = self.h_func(preds)
        true_prob_vectors = true_prob_vectors / true_prob_vectors.sum(axis=1, keepdims=True)
        
        kl_per_sample = np.sum(true_prob_vectors * np.log(true_prob_vectors / preds), axis=1)
        average_kl = np.mean(kl_per_sample)
        
        return average_kl
    
    def calculate_true_L2_squared_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        
        true_prob_vectors = self.h_func(preds)
        true_prob_vectors = true_prob_vectors / true_prob_vectors.sum(axis=1, keepdims=True)
        
        diff = preds - true_prob_vectors
        vector_lz_errors = np.linalg.norm(diff, ord=2, axis=1, keepdims=True)**2
        true_vector_ece = np.mean(vector_lz_errors)
        
        return true_vector_ece
    

class BinaryCalibrationSimulator:
    def __init__(self, h_func, style='uniform', beta_params=(0.5, 0.5)):
        self.h_func = h_func
        self.style = style
        self.beta_params = beta_params
    
    def generate_preds(self, n_samples=1000):
        if self.style == 'uniform':
            preds = np.random.uniform(0, 1, n_samples)
        elif self.style == 'beta':
            preds = np.random.beta(self.beta_params[0], self.beta_params[1], n_samples)
        else:
            raise ValueError("Style must be 'uniform' or 'beta'")
        
        return preds

    def generate_labels(self, preds):
        true_probs = self.h_func(preds)
        true_probs = np.clip(true_probs, 0, 1)
        
        labels = np.random.binomial(1, true_probs)
        
        return labels

    def calculate_true_L1_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        absolute_errors = np.abs( preds - self.h_func(preds) )
        true_ece = np.mean(absolute_errors)
        
        return true_ece
    
    def calculate_true_KL_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)

        p = self.h_func(preds)
        q = preds
        kl_samples = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        # kl_per_sample = np.sum(self.h_func(preds) * np.log(self.h_func(preds) / preds))
        average_kl = np.mean(kl_samples)

        return average_kl
    
    def calculate_true_L2_squared_ece(self, n_samples=100_000):
        preds = self.generate_preds(n_samples=n_samples)
        absolute_errors = np.abs( preds - self.h_func(preds) )**2
        true_ece = np.mean(absolute_errors)
        
        return true_ece

def perfectly_calibrated_bin(p):
    return p

def almost_perfectly_calibrated_binary(p, eps=0.02):
    return np.clip(p + eps, 0, 1)

def overconfident_binary(p):
    logits  = invert_sigmoid(p)
    return sigmoid( logits * 0.4 + 0.3 )

def underconfident_binary(p):
    logits  = invert_sigmoid(p)
    return sigmoid( logits * 4 + 1 )

def perfectly_calibrated_mc(preds):
    return preds

def overconfident_mc(preds, T=0.3):
    logits = multiclass_probs_to_logits(preds)
    return softmax(logits * T) 

def underconfident_mc(preds, T=2):
    logits = multiclass_probs_to_logits(preds)
    return softmax(logits * T) 

def underconfident_and_shifted_mc(preds, T=2, shift = 2):
    logits = multiclass_probs_to_logits(preds)
    return softmax(logits * T + 2) 

def harmonic_distortion_mc(preds, frequency=5.0, amplitude=0.5):
    logits = multiclass_probs_to_logits(preds)
    ripple = np.sin(logits * frequency) * amplitude
    return softmax(logits + ripple)

def evaluate_ece_bin_binary(p, y, n_bins=15, random_state=42):
    n_samples, n_classes = p.shape
    
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1

    kmeans = KMeans(n_clusters=n_bins, random_state=random_state, n_init='auto')
    bin_ids = kmeans.fit_predict(p)
    
    ece = 0.0

    for b in range(n_bins):
        mask = (bin_ids == b)
        bin_size = np.sum(mask)
        
        if bin_size > 0:
            conf_b = np.mean(p[mask], axis=0)
            acc_b = np.mean(y_one_hot[mask], axis=0)
            dist = np.linalg.norm(conf_b - acc_b, ord=z)
            ece += (bin_size / n_samples) * dist
            
    return ece

def evaluate_ece_bin_1d(p, y, n_bins=15):
    n_samples = p.shape[0]
    
    p = p.flatten()
    y = y.flatten()
    
    indices = np.argsort(p)
    p_sorted = p[indices]
    y_sorted = y[indices]
    
    p_bins = np.array_split(p_sorted, n_bins)
    y_bins = np.array_split(y_sorted, n_bins)
    
    ece = 0.0
    
    for b_p, b_y in zip(p_bins, y_bins):
        bin_size = len(b_p)
        if bin_size > 0:
            conf_b = np.mean(b_p)
            acc_b = np.mean(b_y)
            dist = np.abs(conf_b - acc_b)
            ece += (bin_size / n_samples) * dist
            
    return ece

def evaluate_ece_bin(p, y, z=1, n_bins=15, random_state=42):
    n_samples, n_classes = p.shape
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    
    kmeans = KMeans(n_clusters=n_bins, random_state=random_state, n_init='auto')
    bin_ids = kmeans.fit_predict(p)
    
    ece = 0.0
    
    for b in range(n_bins):
        mask = (bin_ids == b)
        bin_size = np.sum(mask)
        
        if bin_size > 0:
            conf_b = np.mean(p[mask], axis=0)
            acc_b = np.mean(y_one_hot[mask], axis=0)
            dist = np.linalg.norm(conf_b - acc_b, ord=z)
            ece += (bin_size / n_samples) * dist
            
    return ece


def wavy_increasing_function_bin(x, waves=[5, 10], intensity=[2, 1]):
    k = waves[0] * 2 
    k2 = waves[1] * 2 
    amplitude = (intensity[0] / (k * np.pi))
    amplitude2 = (intensity[1] / (k * np.pi))
    
    return x + amplitude * np.sin(k * np.pi * x) + amplitude2 * np.sin(k2 * (np.pi-np.pi/3) * x)
