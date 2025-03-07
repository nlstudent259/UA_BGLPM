# Blood Glucose Forecasting with Uncertainty Quantification

This repository implements a **Uncertainty-Aware Blood Glucose Prediction Model**.

Key Features

✅ Bi-GRU:
Captures sequential and temporal patterns in glucose data.

✅ Attention Mechanism:
Focuses on important time steps for prediction.

✅ Bayesian Linear Output Layer:
Predicts both the mean and variance of blood glucose levels, modeling aleatoric uncertainty (inherent data noise).

✅ MC Dropout:
Applies Monte Carlo dropout during inference to estimate epistemic uncertainty (model uncertainty), by performing multiple stochastic forward passes.

✅ Uncertainty Decomposition:

Epistemic Uncertainty: Reflects the model’s confidence in its own parameters.

Aleatoric Uncertainty: Captures noise and variability in the data itself.

✅ Comprehensive Uncertainty Evaluation:

Calibration analysis,
Prediction interval coverage,
Uncertainty correlation with absolute error

✅ Result Visualization:
Clear plots for predictions, uncertainty bands, and performance metrics.

---

## Data
Data should follow the format:
| timestamp | glucose | 
|---|---|
| 2024-01-01 08:00 | 120 | 
| ... | ... | 

Place files in `/Data/`.

---

## Installation

pip install -r requirements.txt

## Running Training and Validation

python main.py
