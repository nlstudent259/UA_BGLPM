import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

# evidential loss for evidential model training
def evidential_loss(y_true, gamma, v, alpha, beta, lambda_reg=0.01):
    diff = y_true - gamma
    
    # NLL from NIG distribution
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(2 * beta) \
        + (alpha + 0.5) * torch.log(v * diff**2 + 2 * beta) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    
    # Regularization: penalize overconfidence when error is high
    reg = torch.abs(diff) * (2 * v + alpha)
    
    # Total loss
    loss = nll + lambda_reg * reg
    
    return loss.mean()

def bnn_loss(model, y_true, mean_pred, var_pred):
    std_pred = torch.sqrt(var_pred + 1e-6)
    error = torch.abs(y_true - mean_pred)
    squared_error = error**2
    
    nll_loss = 0.5 * torch.mean(torch.log(var_pred) + squared_error / var_pred)+ model.kl_divergence() * 0.01 #negative log likelihood
    return nll_loss

# Training function 
def train_bnn(model, train_loader, test_loader, epochs, lr, device, modelEv=False, scheduler=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    if scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)  # Reduce LR if loss plateaus
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')  # Initialize the best test loss as infinity
    epochs_since_improvement = 0  # Counter to track the number of epochs without improvement
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for X_batch, y_batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if modelEv:
                gamma, v, alpha, beta = model(X_batch)
                loss = evidential_loss(y_batch, gamma, v, alpha, beta)
            else:
                mean_pred, var_pred = model(X_batch)
                loss = bnn_loss(model, y_batch, mean_pred, var_pred) 
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_loss =  epoch_train_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        epoch_test_loss = 0.0
        batch_errors = []
        batch_uncertainties = []
        with torch.no_grad():
            
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)     
                
                if modelEv:
                    gamma, v, alpha, beta = model(X_test)
                    loss = evidential_loss(y_test, gamma, v, alpha, beta)
                else:
                    mean_pred, var_pred = model(X_test)
                    loss = bnn_loss(model, y_test, mean_pred, var_pred) 
                
                epoch_test_loss += loss.item()
        
        test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(test_loss)
        
        if scheduler:
            scheduler.step(test_loss)  # Update the learning rate based on loss
            
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        # Early stopping logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss  # Update the best test loss
            epochs_since_improvement = 0  # Reset the counter
            torch.save(model.state_dict(), f'Models/bestmodel_{epoch}_GPU2.pth')
            print(f"✅ Model saved with best test loss: {best_test_loss:.4f}")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= 20:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in test loss.")
            break  # Stop training if no improvement for 'patience' epochs
    return train_losses, test_losses
