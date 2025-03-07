import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Normalization function
def normalize(data, norm_param=None):
    if norm_param is None:
        mean_val, std_val = np.mean(data), np.std(data)
        norm_param = {'mean': mean_val, 'std': std_val}
    else:
        mean_val, std_val = norm_param['mean'], norm_param['std']
    return (data - mean_val) / std_val, norm_param

# Denormalization function
def denormalize(normalized_data, norm_param, BGL):
    # Retrieve normalization parameters
    mean_val, std_val = norm_param['mean'], norm_param['std']
    
    normalized_data = normalized_data.cpu().detach().numpy()  # Convert tensor to numpy array
    # Denormalize 
    denormalized = (normalized_data * std_val) + mean_val 
  
    return denormalized #denormalized_data

# Create sequences
def createsequences(data, window_size, step_size):
    X, y = [], []
    for i in range(len(data) - window_size - step_size + 1):
        X.append(data[i:i + window_size, :])
        y.append(data[i + window_size + step_size - 1, 0])
    return np.array(X), np.array(y)

# Data Loader
def load_data(path, window_size, step_size, batch_size, norm_param=None):
    raw_df = pd.read_csv(path)
    glucose = raw_df['glucose'].values.astype(np.float32).reshape(-1, 1)
    data_normalized, norm_param = normalize(glucose, norm_param)
    X, y = createsequences(data_normalized, window_size, step_size)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    return loader, norm_param

# for calibration analysis
def compute_coverage(y_true, y_pred, aleatoric, epistemic):
    total_uncertainty = aleatoric + epistemic
    lower = y_pred - 1.96 * total_uncertainty
    upper = y_pred + 1.96 * total_uncertainty
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)

