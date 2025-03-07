import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#uncertainty calibration
def evaluate_uncertainty_calibration(y_true, mean_pred, std_pred):

    lower_bound = mean_pred - 2 * std_pred
    upper_bound = mean_pred + 2 * std_pred
    
    within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    calibration_percentage = np.mean(within_interval) * 100  # Convert to percentage
    
    return calibration_percentage

# evaluate
def evaluate(model, test_loader, device, norm_param, num_samples=100):
    model.eval()  # Set to evaluation mode, but dropout will still be applied
    
    all_y_test = []
    all_means = []
    all_epistemic = []
    all_aleatoric = []
    all_total_uncertainty = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            all_y_test.append(y_test.cpu().numpy())
            
            # Use MC dropout to get predictions and uncertainty estimates
            mean_pred, epistemic, aleatoric, total = model.mc_dropout_predict(X_test, num_samples)
            
            all_means.append(mean_pred.cpu().numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
            all_total_uncertainty.append(total.cpu().numpy())
    
    # Concatenate results
    y_test_all = np.concatenate(all_y_test)
    mean_preds_all = np.concatenate(all_means)
    epistemic_uncertainty = np.concatenate(all_epistemic)
    aleatoric_uncertainty = np.concatenate(all_aleatoric)
    total_uncertainty = np.concatenate(all_total_uncertainty)
    
    # Convert to standard deviations
    epistemic_std = np.sqrt(epistemic_uncertainty)
    aleatoric_std = np.sqrt(aleatoric_uncertainty)
    total_std = np.sqrt(total_uncertainty)
    
    # Calculate calibration
    calibration_score = evaluate_uncertainty_calibration(
        y_test_all, mean_preds_all, total_std)
    
    # Denormalize for visualization and analysis
    y_test_denorm = (y_test_all * norm_param['std']) + norm_param['mean']
    mean_pred_denorm = (mean_preds_all * norm_param['std']) + norm_param['mean']
    total_std_denorm = total_std * norm_param['std']
    epistemic_std_denorm = epistemic_std * norm_param['std']
    aleatoric_std_denorm = aleatoric_std * norm_param['std']

    # Compute MSE and RMSE
    mse = np.mean((mean_pred_denorm - y_test_denorm) ** 2)
    rmse = np.sqrt(mse)
    
    return {
        'y_true': y_test_denorm,
        'mean_pred': mean_pred_denorm,
        'total_std': total_std_denorm,
        'epistemic_std': epistemic_std_denorm,
        'aleatoric_std': aleatoric_std_denorm,
        'calibration_score': calibration_score,
        'mse': mse,
        'rmse': rmse
    }

def evaluate_ev(model, loader, device,norm_param):
    model.eval()
    all_y, all_gamma, all_v, all_alpha, all_beta = [], [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            gamma, v, alpha, beta = model(X_batch)

            all_y.extend(y_batch.cpu().numpy())
            all_gamma.extend(gamma.cpu().numpy())
            all_v.extend(v.cpu().numpy())
            all_alpha.extend(alpha.cpu().numpy())
            all_beta.extend(beta.cpu().numpy())

    all_y = np.array(all_y)
    all_gamma = np.array(all_gamma)
    all_v = np.array(all_v)
    all_alpha = np.array(all_alpha)
    all_beta = np.array(all_beta)

    # Calculate uncertainties
    # Aleatoric uncertainty (from the data)
    aleatoric = np.sqrt(all_beta / (all_alpha - 1))
    
    # Epistemic uncertainty (from model/lack of data)
    epistemic = np.sqrt(all_beta / (all_v * (all_alpha - 1)))
    
    # Total predictive uncertainty
    total_uncertainty = np.sqrt(all_beta * (1 + 1/all_v) / (all_alpha - 1))

    
    # Absolute errors
    errors = np.abs(all_y - all_gamma)

    
    lower_bound = all_gamma - 2 * total_uncertainty
    upper_bound = all_gamma + 2 * total_uncertainty
    
    within_interval = (all_y >= lower_bound) & (all_y <= upper_bound)
    calibration_percentage = np.mean(within_interval) * 100  # Convert to percentage
    print(f"Uncertainty Calibration: {calibration_percentage}% (Expected: ~95%)")
    
    # Error-uncertainty correlation
    correlation = np.corrcoef(errors, total_uncertainty)[0, 1]
    print(f"Error-Uncertainty Correlation: {correlation:.3f}")
    

    # Denormalize for visualization and analysis
    y_test_denorm = (all_y * norm_param['std']) + norm_param['mean']
    mean_pred_denorm = (all_gamma * norm_param['std']) + norm_param['mean']
    total_std_denorm = total_uncertainty * norm_param['std']
    epistemic_std_denorm = epistemic * norm_param['std']
    aleatoric_std_denorm = aleatoric * norm_param['std']

    # Compute MSE and RMSE
    mse = np.mean((mean_pred_denorm - y_test_denorm) ** 2)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.3f}")
    

    # Plot: True vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm[:288*2], label="True Values", color="blue", alpha=0.6)
    plt.plot(mean_pred_denorm[:288*2], label="Predicted Mean", color="orange", alpha=0.8)
    
    # Add alert thresholds
    plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Hypoglycemia Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', alpha=0.7, label='Hyperglycemia Threshold')    
    plt.xlabel("Time Step")
    plt.ylabel("Glucose Level")
    plt.legend()
    plt.savefig("TruevsPrediction.png", dpi=300, bbox_inches="tight")
    plt.show()
    

    # Plot error vs uncertainty
    plt.figure(figsize=(10, 5))
    plt.scatter(errors, total_uncertainty, alpha=0.5)
    plt.xlabel("Absolute Error")
    plt.ylabel("Total Predicted Uncertainty")
    plt.grid(True)
    
    # Add diagonal line for perfect calibration
    max_val = max(errors.max(), total_uncertainty.max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.savefig('error_vs_uncertainty.png', dpi=600)
    plt.show()

    return {
        'y_true': y_test_denorm,
        'mean_pred': mean_pred_denorm,
        'total_std': total_std_denorm,
        'epistemic_std': epistemic_std_denorm,
        'aleatoric_std': aleatoric_std_denorm,
        'calibration_score': calibration_percentage,
        'mse': mse,
        'rmse': rmse
    }
