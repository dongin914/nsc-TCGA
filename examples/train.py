import sys
sys.path.append('../')
sys.path.append('../DeepSurvivalMachines/')
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from nsc import NeuralSurvivalCluster
import gc
import pandas as pd
from sklearn.model_selection import ParameterSampler

def train_model(x_train, t_train_ddh, e_train, x_dev, t_dev_ddh, e_dev, x_val, t_val_ddh, e_val, params):
    models = []
    for param in params:
        model = NeuralSurvivalCluster(layers=param['layers'], act=param['act'], k=param['k'],
                                      layers_surv=param['layers_surv'], representation=param['representation'])
        model.fit(x_train, t_train_ddh, e_train, n_iter=10, bs=param['batch'],
                  lr=param['learning_rate'], val_data=(x_dev, t_dev_ddh, e_dev))
        nll = model.compute_nll(x_val, t_val_ddh, e_val)
        if not(np.isnan(nll)):
            models.append([nll, model])
        else:
            print("WARNING: Nan Value Observed")

    best_model = min(models, key=lambda x: x[0])
    model = best_model[1]
    torch.save(model, "./model/model_weight.pth")
    return model
