import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def analyze_uncertainty_clinical_relevance(y_test_denorm, mean_pred_denorm, total_std_denorm, 
                                          epistemic_std_denorm, aleatoric_std_denorm, norm_param):
   
    # 1. Error vs. Uncertainty Analysis
    print("\n========== ERROR VS UNCERTAINTY ANALYSIS ==========")
    
    # Calculate absolute prediction errors
    abs_errors = np.abs(y_test_denorm - mean_pred_denorm)    
    # Use variance instead of standard deviation for consistency
    total_std_denorm = total_std_denorm
    
    # Create a scatter plot of absolute error vs. total uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(total_std_denorm, abs_errors, alpha=0.5, color='blue')
    plt.xlabel('Prediction Uncertainty (Total Std)')
    plt.ylabel('Absolute Prediction Error')
    plt.grid(alpha=0.3)    
    # Add a trend line
    z = np.polyfit(total_std_denorm, abs_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(total_std_denorm), p(np.sort(total_std_denorm)), 
             "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")    
    # Calculate Pearson correlation
    correlation = np.corrcoef(total_std_denorm, abs_errors)[0,1]
    plt.annotate(f"Correlation: {correlation:.2f}", 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))    
    plt.legend()
    plt.savefig("Results/Error_vs_Uncertainty.png", dpi=600, bbox_inches="tight")
    plt.show()

    #####################################################################################
    
    
    print(f"Correlation between prediction error and uncertainty: {correlation:.2f}")
    print(f"Linear relationship: Error = {z[0]:.2f} × Uncertainty + {z[1]:.2f}")
    
    # Create a grouped bar chart for different uncertainty levels
    # Define uncertainty quantiles
    q_25, q_50, q_75 = np.percentile(total_std_denorm, [25, 50, 75])
    uncertainty_groups = ['Low', 'Medium-Low', 'Medium-High', 'High']
    
    # Calculate mean error for each uncertainty group
    mask_low = total_std_denorm <= q_25
    mask_med_low = (total_std_denorm > q_25) & (total_std_denorm <= q_50)
    mask_med_high = (total_std_denorm > q_50) & (total_std_denorm <= q_75)
    mask_high = total_std_denorm > q_75
    
    mean_errors = [
        np.mean(abs_errors[mask_low]),
        np.mean(abs_errors[mask_med_low]),
        np.mean(abs_errors[mask_med_high]),
        np.mean(abs_errors[mask_high])
    ]
    
    # Plot mean error by uncertainty group
    plt.figure(figsize=(10, 6))    
    # Create the bars
    bars = plt.bar(uncertainty_groups, mean_errors, color=['lightblue', 'skyblue', 'royalblue', 'darkblue'])    
    # Add value labels on top of bars - with adjusted positioning
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 0.8,  # Position inside the bar near the top
                 f'{height:.1f}', ha='center', va='top',
                 color='white', fontweight='bold')  # White text for better contrast    
    # Set y-axis limit with some padding above the highest bar
    max_height = max(mean_errors)
    plt.ylim(0, max_height * 1.1)  # Add 10% padding above the highest bar    
    # Enhance the plot with better styling
    plt.xlabel('Uncertainty Level', fontsize=12)
    plt.ylabel('Mean Absolute Error (mg/dL)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()  # Adjust layout to make sure everything fits    
    # Save the figure
    plt.savefig("Results/Error_by_Uncertainty_Group_test.png", dpi=300, bbox_inches="tight")
    plt.show()

    ################################################################################################
    
    
    # Calculate error quantiles for specific points
    high_uncertainty_points = abs_errors[mask_high]
    low_uncertainty_points = abs_errors[mask_low]
    
    print(f"Mean error for high uncertainty points: {np.mean(high_uncertainty_points):.2f} mg/dL")
    print(f"Mean error for low uncertainty points: {np.mean(low_uncertainty_points):.2f} mg/dL")
    print(f"Ratio of high to low uncertainty errors: {np.mean(high_uncertainty_points)/np.mean(low_uncertainty_points):.2f}x")
    
    # 2. Clinical Relevance Analysis
    print("\n========== CLINICAL RELEVANCE ANALYSIS ==========")
    
    # Define glucose ranges based on clinical importance
    hypoglycemia_threshold = 70  # mg/dL, common threshold for low blood sugar
    hyperglycemia_threshold = 180  # mg/dL, common threshold for high blood sugar
    
    # Create masks for different glucose ranges
    mask_hypo = y_test_denorm < hypoglycemia_threshold
    mask_normal = (y_test_denorm >= hypoglycemia_threshold) & (y_test_denorm <= hyperglycemia_threshold)
    mask_hyper = y_test_denorm > hyperglycemia_threshold
    
    # Calculate percentage of samples in each range
    print(f"Percentage of hypoglycemic samples: {np.mean(mask_hypo)*100:.1f}%")
    print(f"Percentage of normal samples: {np.mean(mask_normal)*100:.1f}%")
    print(f"Percentage of hyperglycemic samples: {np.mean(mask_hyper)*100:.1f}%")
    
    # Calculate mean uncertainties for each range
    mean_total_unc = [
        np.mean(total_std_denorm[mask_hypo]) if np.any(mask_hypo) else 0,
        np.mean(total_std_denorm[mask_normal]) if np.any(mask_normal) else 0,
        np.mean(total_std_denorm[mask_hyper]) if np.any(mask_hyper) else 0
    ]
    
    mean_epistemic_unc = [
        np.mean(epistemic_std_denorm[mask_hypo]) if np.any(mask_hypo) else 0,
        np.mean(epistemic_std_denorm[mask_normal]) if np.any(mask_normal) else 0,
        np.mean(epistemic_std_denorm[mask_hyper]) if np.any(mask_hyper) else 0
    ]
    
    mean_aleatoric_unc = [
        np.mean(aleatoric_std_denorm[mask_hypo]) if np.any(mask_hypo) else 0,
        np.mean(aleatoric_std_denorm[mask_normal]) if np.any(mask_normal) else 0,
        np.mean(aleatoric_std_denorm[mask_hyper]) if np.any(mask_hyper) else 0
    ]
    
    # Plot mean uncertainty by clinical glucose range
    glucose_ranges = ['Hypoglycemia\n(<70 mg/dL)', 'Normal\n(70-180 mg/dL)', 'Hyperglycemia\n(>180 mg/dL)']
    
    plt.figure(figsize=(12, 7))
    x = np.arange(len(glucose_ranges))
    width = 0.25    
    plt.bar(x - width, mean_total_unc, width, label='Total', color='red', alpha=0.7)
    plt.bar(x, mean_epistemic_unc, width, label='Epistemic', color='green', alpha=0.7)
    plt.bar(x + width, mean_aleatoric_unc, width, label='Aleatoric', color='purple', alpha=0.7)    
    plt.ylabel('Mean Standard Deviation (mg/dL)')
    plt.xlabel('Glucose Range')
    plt.xticks(x, glucose_ranges)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("Results/Uncertainty_by_Clinical_Range.png", dpi=300, bbox_inches="tight")
    plt.show()

    ############################################################################
    
    
    # Print detailed uncertainty by range
    for i, range_name in enumerate(glucose_ranges):
        print(f"\n{range_name}:")
        print(f"  Total uncertainty: {mean_total_unc[i]:.2f} mg/dL")
        print(f"  Epistemic uncertainty: {mean_epistemic_unc[i]:.2f} mg/dL")
        print(f"  Aleatoric uncertainty: {mean_aleatoric_unc[i]:.2f} mg/dL")
        print(f"  Epistemic/Aleatoric ratio: {mean_epistemic_unc[i]/mean_aleatoric_unc[i]:.2f}" if mean_aleatoric_unc[i] > 0 else "  Epistemic/Aleatoric ratio: N/A")
    
    # Analyze "safety margin" provided by uncertainty bounds
    # Calculate percentage of times where actual value falls outside predicted mean but inside uncertainty bounds
    outside_mean_inside_bounds = ((y_test_denorm > mean_pred_denorm) & 
                                  (y_test_denorm <= mean_pred_denorm + 2 * total_std_denorm)) | \
                                 ((y_test_denorm < mean_pred_denorm) & 
                                  (y_test_denorm >= mean_pred_denorm - 2 * total_std_denorm))
    
    safety_margin_percentage = np.mean(outside_mean_inside_bounds) * 100
    
    print(f"\nSafety margin percentage: {safety_margin_percentage:.2f}% of points fall outside the predicted mean")
    print(f"but within uncertainty bounds, providing a 'safety margin' for clinical decisions.")
    
    # Calculate "false alarm rate" - where uncertainty bounds are large but prediction is accurate
    close_prediction = np.abs(y_test_denorm - mean_pred_denorm) < 10  # within 10 mg/dL
    high_uncertainty = total_std_denorm > np.percentile(total_std_denorm, 75)  # top 25% uncertainty
    
    false_alarm_rate = np.mean(close_prediction & high_uncertainty) * 100
    print(f"False alarm rate: {false_alarm_rate:.2f}% of predictions are accurate but have high uncertainty")
    
    ############################################################################
    
    
    # 3. Analyze temporal patterns in uncertainty
    print("\n========== TEMPORAL PATTERNS ANALYSIS ==========")
    
    # Calculate rate of change in true glucose values
    glucose_rate_of_change = np.diff(y_test_denorm.flatten(), prepend=y_test_denorm.flatten()[0])
    abs_glucose_rate = np.abs(glucose_rate_of_change)
    
    # Plot uncertainty vs absolute rate of change
    plt.figure(figsize=(14, 8))
    plt.scatter(abs_glucose_rate, total_std_denorm, alpha=0.4, color='blue')
    plt.xlabel('Absolute Glucose Rate of Change (mg/dL/step)')
    plt.ylabel('Prediction Uncertainty (Total Std)')    
    # Add a trend line
    z = np.polyfit(abs_glucose_rate, total_std_denorm, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(abs_glucose_rate), p(np.sort(abs_glucose_rate)), 
             "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")    
    # Calculate Pearson correlation
    roc_correlation = np.corrcoef(abs_glucose_rate, total_std_denorm)[0,1]
    plt.annotate(f"Correlation: {roc_correlation:.2f}", 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("Results/Uncertainty_vs_Glucose_RoC.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Correlation between glucose rate of change and uncertainty: {roc_correlation:.2f}")
    print(f"This suggests that {'rapid changes in glucose levels correspond to higher prediction uncertainty' if roc_correlation > 0.3 else 'uncertainty is not strongly affected by the rate of glucose change'}")
    
    return

def BasicAnalysis(y_test_denorm, mean_pred_denorm, total_std_denorm, epistemic_std_denorm, aleatoric_std_denorm, norm_param):    
    # Plot 1: True vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm[:288*2], label="True Values", color="blue", alpha=0.6)
    plt.plot(mean_pred_denorm[:288*2], label="Predicted Mean", color="orange", alpha=0.8) 
    plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Hypoglycemia Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', alpha=0.7, label='Hyperglycemia Threshold')    
    plt.xlabel("Time Step")
    plt.ylabel("Glucose Level")
    plt.legend()
    plt.savefig("Results/TruevsPrediction.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Plot 2: Total Uncertainty
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm, label="True Values", color="blue", alpha=0.6)
    plt.plot(mean_pred_denorm, label="Predicted Mean", color="orange", alpha=0.8)
    plt.fill_between(range(len(mean_pred_denorm)), 
                    mean_pred_denorm - 2 * total_std_denorm, 
                    mean_pred_denorm + 2 * total_std_denorm, 
                    color="red", alpha=0.3, label="95% Confidence Interval")
    plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Hypoglycemia Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', alpha=0.7, label='Hyperglycemia Threshold')    
    plt.xlabel("Time Step")
    plt.ylabel("Glucose Level")
    plt.legend()
    plt.savefig("Results/Glucose_Prediction_Total_Uncertainty.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Plot 3: Epistemic Uncertainty
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm, label="True Values", color="blue", alpha=0.6)
    plt.plot(mean_pred_denorm, label="Predicted Mean", color="orange", alpha=0.8)
    plt.fill_between(range(len(mean_pred_denorm)), 
                    mean_pred_denorm - 2 * epistemic_std_denorm, 
                    mean_pred_denorm + 2 * epistemic_std_denorm, 
                    color="green", alpha=0.3, label="Epistemic Uncertainty (±2σ)")
    plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Hypoglycemia Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', alpha=0.7, label='Hyperglycemia Threshold')    
    plt.xlabel("Time Step")
    plt.ylabel("Glucose Level")
    plt.legend()
    plt.savefig("Results/Glucose_Prediction_Epistemic_Uncertainty.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Plot 4: Aleatoric Uncertainty
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm, label="True Values", color="blue", alpha=0.6)
    plt.plot(mean_pred_denorm, label="Predicted Mean", color="orange", alpha=0.8)
    plt.fill_between(range(len(mean_pred_denorm)), 
                    mean_pred_denorm - 2 * aleatoric_std_denorm, 
                    mean_pred_denorm + 2 * aleatoric_std_denorm, 
                    color="purple", alpha=0.3, label="Aleatoric Uncertainty (±2σ)")
    plt.axhline(y=70, color='r', linestyle=':', alpha=0.7, label='Hypoglycemia Threshold')
    plt.axhline(y=180, color='orange', linestyle=':', alpha=0.7, label='Hyperglycemia Threshold')    
    plt.xlabel("Time Step")
    plt.ylabel("Glucose Level")
    plt.legend()
    plt.savefig("Results/Glucose_Prediction_Aleatoric_Uncertainty.png", dpi=300, bbox_inches="tight")
    plt.show()

