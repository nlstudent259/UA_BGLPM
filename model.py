import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

# bayesian Linear Layer with Aleatoric Uncertainty
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization for weights
        weight_std = math.sqrt(2.0 / (in_features + out_features)) #0.1
       
        bias_std = math.sqrt(1.0 / out_features) #0.1
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * weight_std)
        self.bias_mu = nn.Parameter(torch.randn(out_features) * bias_std)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3)
        self.weight_prior, self.bias_prior = Normal(0, 1), Normal(0, 1)
    
    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weights = Normal(self.weight_mu, weight_sigma).rsample()
        biases = Normal(self.bias_mu, bias_sigma).rsample()
        output = x @ weights.T + biases
        mean, log_var = output[:, 0], output[:, 1]  # Mean & Log Variance
        var = F.softplus(log_var) + 1e-6
        return mean, var
    
    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        kl_weights = kl_divergence(Normal(self.weight_mu, weight_sigma), self.weight_prior).sum()
        kl_biases = kl_divergence(Normal(self.bias_mu, bias_sigma), self.bias_prior).sum()
        return kl_weights + kl_biases

# Baysian Neural Network 
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3, mc_dropout=True, mc_dropout_rate=None):
        super().__init__()
        self.mc_dropout = mc_dropout  # enable MC dropout at inference time
        self.dropout_rate = dropout
        self.mc_dropout_rate = mc_dropout_rate if mc_dropout_rate is not None else dropout
        
        # GRU layers with dropout
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
        # attention mechanism
        self.attn = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.context_vector = nn.Linear(2 * hidden_dim, 1, bias=False)
                
        # Dropout layer (to be used in both training and inference when mc_dropout=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.mc_dropout = nn.Dropout(self.mc_dropout_rate)
        # layer normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        # Output layer
        self.fc = BayesianLinear(2*hidden_dim, output_dim * 2)  # Output both mean & variance
        
        # Initialize weights for better convergence
        #if self.training:
         #   self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, use_mc_dropout=False):        
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)

        # attention mechanism
        attn_weights = torch.tanh(self.attn(gru_out))
        attn_weights = self.context_vector(attn_weights).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, gru_out).squeeze(1)
        
        if self.training:
            context = self.dropout(context)
        elif self.mc_dropout and use_mc_dropout:
            context = self.mc_dropout(context)
        
        # Final output
        return self.fc(context)
    
    def kl_divergence(self):
        return self.fc.kl_divergence()
    
    # sample predictions using MC dropout
    def mc_dropout_predict(self, x, num_samples=100):
        self.eval()  # Set model to evaluation mode
        means, variances = [], []
        
        # Perform multiple forward passes with dropout enabled
        for _ in range(num_samples):
            mean, var = self.forward(x, use_mc_dropout=True)
            means.append(mean.unsqueeze(0))
            variances.append(var.unsqueeze(0))
        
        # Stack all predictions
        means = torch.cat(means, dim=0)
        variances = torch.cat(variances, dim=0)
        
        # Calculate total uncertainty components
        # Epistemic uncertainty: variance of the means
        epistemic_uncertainty = torch.var(means, dim=0)
        
        # Aleatoric uncertainty: mean of the predicted variances
        aleatoric_uncertainty = torch.mean(variances, dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Final prediction: mean of means
        final_mean = torch.mean(means, dim=0)
        
        return final_mean, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty

#evidential learning model
class EvidentialGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=True
        )
        self.attn = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.context_vector = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.fc = nn.Linear(2 * hidden_dim, 4)  # Outputs γ, v, α, β
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x):
        # Run through GRU
        gru_out, _ = self.gru(x)  # [batch, seq_len, 2*hidden_dim]
        
        # Attention mechanism
        attn_weights = torch.tanh(self.attn(gru_out))  # [batch, seq_len, 2*hidden_dim]
        attn_scores = self.context_vector(attn_weights)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        
        
        context = torch.bmm(attn_weights.transpose(1, 2), gru_out)  # [batch, 1, 2*hidden_dim]
        context = context.squeeze(1)  # [batch, 2*hidden_dim]

        # evidential parameters
        evidential_params = self.fc(context)
        
        # activations
        gamma = evidential_params[:, 0]          # Mean (Prediction) - no activation
        v = F.softplus(evidential_params[:, 1]) + 1e-6  # Precision - must be positive
        alpha = F.softplus(evidential_params[:, 2]) + 1.0  # Shape parameter - must be > 1
        beta = F.softplus(evidential_params[:, 3]) + 1e-6  # Scale parameter - must be positive

        return gamma, v, alpha, beta
