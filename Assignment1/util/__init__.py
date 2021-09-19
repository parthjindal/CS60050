"""
Utility module for loading dataset and other functions.    
"""
from .dataset import CarDataset
from .utility import information_gain, \
    gini_gain, entropy, gini_index, shuffle_dataset, \
    split_dataset, seed_everything, chi2_contingency
from .metrics import get_metrics, find_ci_interval