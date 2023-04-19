import sys
sys.path.append('../')
sys.path.append('../DeepSurvivalMachines/')

import numpy as np
import torch
from train import preprocess_data, train_nsc_models
from eval import evaluate_nsc_models, plot_survival_clusters

np.random.seed(42)
torch.random.manual_seed(42)

# Load data
x = np.load('../ProcessedData/GeneCount.npy', allow_pickle=True)
t = np.load('../ProcessedData/TTE.npy', allow_pickle=True)
e = np.load('../ProcessedData/Event.npy', allow_pickle=True)

# Preprocess and split data
x_train, x_test, t_train, t_test, e_train, e_test, x_dev, x_val, t_dev, t_val, e_dev, e_val, times_ddh = preprocess_data(x, t, e)

# Train models and select the best one
best_model = train_nsc_models(x_train, t_train, e_train, x_dev, t_dev, e_dev, x_val, t_val, e_val)

# Evaluate the best model
evaluate_nsc_models(best_model, x_test, t_test, e_test, t_train, e_train, times_ddh)

# Plot survival clusters
plot_survival_clusters(best_model, t)
