import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterSampler
from nsc import NeuralSurvivalCluster
import gc
import torch

def preprocess_data(x, t, e):
    horizons = [0.5]
    times = np.quantile(t[e != 0], horizons).tolist()
    
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(x, t, e, test_size=0.2, random_state=42)
    x_train, x_val, t_train, t_val, e_train, e_val = train_test_split(x_train, t_train, e_train, test_size=0.2, random_state=42)
    x_dev, x_val, t_dev, t_val, e_dev, e_val = train_test_split(x_val, t_val, e_val, test_size=0.5, random_state=42)

    minmax = lambda x: x / t_train.max()  # Enforce to be inferior to 1
    t_train_ddh = minmax(t_train)
    t_dev_ddh = minmax(t_dev)
    t_val_ddh = minmax(t_val)
    times_ddh = minmax(np.array(times))

    return x_train, x_test, t_train, t_test, e_train, e_test, x_dev, x_val, t_dev, t_val, e_dev, e_val, times_ddh

def train_nsc_models(x_train, t_train, e_train, x_dev, t_dev, e_dev, x_val, t_val, e_val):
    layers = [[50], [50, 50], [50, 50, 50], [100], [100, 100], [100, 100, 100]]
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'layers_surv': layers,
        'k': [2],
        'representation': [50, 100],
        'layers': layers,
        'act': ['Tanh'],
        'batch': [100, 250],
    }
   
params = ParameterSampler(param_grid, 10, random_state=42)
gc.collect()
torch.cuda.empty_cache()
models = []

for param in params:
    model = NeuralSurvivalCluster(layers=param['layers'], act=param['act'], k=param['k'],
                                  layers_surv=param['layers_surv'], representation=param['representation'])

    model.fit(x_train, t_train, e_train, n_iter=10, bs=param['batch'],
              lr=param['learning_rate'], val_data=(x_dev, t_dev, e_dev))

    nll = model.compute_nll(x_val, t_val, e_val)
    if not (np.isnan(nll)):
        models.append([nll, model])
    else:
        print("WARNING: Nan Value Observed")

best_model = min(models, key=lambda x: x[0])
return best_model[1]
