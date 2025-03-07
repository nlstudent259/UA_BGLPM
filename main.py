import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import sys
from model import BayesianNN,EvidentialGRU
from utils import load_data
from train import train_bnn
from evaluate import evaluate,evaluate_ev
from plots import BasicAnalysis, analyze_uncertainty_clinical_relevance


def main(Train = False, modelEv = False):    
    gpu_id = 2  # for specific GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES to {gpu_id}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")
        
    # Parameters
    batch_size = 16
    window_size = 24
    step_size = 6
    hidden_dim = 50
    epochs = 100
    learning_rate = 0.0001    

    #Load Data
    train_loader, norm_param = load_data('Data/Ohio_TrainSet.csv', window_size, step_size, batch_size)
    valid_loader, _ = load_data('Data/Ohio_ValidSet.csv', window_size, step_size, batch_size, norm_param)
    test_loader, _ = load_data('Data/Ohio_TestSet.csv', window_size, step_size, batch_size, norm_param)

    if modelEv: # Separately implemeted Evidential learning     
        model = EvidentialGRU(input_dim=1, hidden_dim=100).to(device)
    else: # Bayesian NN
        model = BayesianNN(input_dim=1, hidden_dim=hidden_dim, output_dim=1, dropout=0.3, mc_dropout=True).to(device) # Enable MC dropout   
    
    print(model)
    if Train:
        train_bnn(model, train_loader, valid_loader, epochs, learning_rate,device,modelEv=modelEv,scheduler=True)
    else:
        model_path = 'Models1/bestmodel_78_GPU2.pth'  
        model.load_state_dict(torch.load(model_path)) 
    
        # Evaluate 
        if modelEv: # For evidential learning
            results = evaluate_ev(model, test_loader, device,norm_param)
        else:    
            results = evaluate(model, test_loader, device,norm_param, num_samples=200) # MC samples
        
        # Print results
        print(f"Uncertainty Calibration: {results['calibration_score']:.2f}% (Expected: ~95%)")
        print(f"MSE: {results['mse']:.2f}, RMSE: {results['rmse']:.2f}")
        
        # Run analysis functions
        print("\n===== BASIC UNCERTAINTY ANALYSIS =====")
    
        BasicAnalysis(
            results['y_true'], 
            results['mean_pred'], 
            results['total_std'],
            results['epistemic_std'], 
            results['aleatoric_std'], 
            norm_param
        )
        
        print("\n===== ADVANCED UNCERTAINTY ANALYSIS =====")
        analyze_uncertainty_clinical_relevance(
            results['y_true'],
            results['mean_pred'],
            results['total_std'],
            results['epistemic_std'],
            results['aleatoric_std'],
            norm_param
        )


if __name__ == "__main__":
    main()
