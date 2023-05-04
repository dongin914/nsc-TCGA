import numpy as np
import torch
from sklearn.model_selection import train_test_split
from train import train_model
from eval import evaluate, plot_clusters
from sklearn.model_selection import ParameterSampler

x = np.load('../ProcessedData/GeneCount.npy', allow_pickle=True)
t = np.load('../ProcessedData/TTE.npy', allow_pickle=True)
e = np.load('../ProcessedData/Event.npy', allow_pickle=True)

np.random.seed(42)
torch.random.manual_seed(42)

horizons = [0.75]
times = np.quantile(t[e != 0], horizons).tolist()

x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(x, t, e, test_size=0.2, random_state=42)
x_train, x_val, t_train, t_val, e_train, e_val = train_test_split(x_train, t_train, e_train, test_size=0.2, random_state=42)
x_dev, x_val, t_dev, t_val, e_dev, e_val = train_test_split(x_val, t_val, e_val, test_size=0.5, random_state=42)

minmax = lambda x: x / t_train.max()
t_train_ddh = minmax(t_train)
t_dev_ddh = minmax(t_dev)
t_val_ddh = minmax(t_val)
times_ddh = minmax(np.array(times))

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
params = ParameterSampler(param_grid, 280, random_state=42)

model = train_model(x_train, t_train_ddh, e_train, x_dev, t_dev_ddh, e_dev, x_val, t_val_ddh, e_val, params)

et_train = np.array([(e_train[i] == 1, t_train[i]) for i in range(len(e_train))],
                    dtype=[('e', bool), ('t', float)])
et_test = np.array([(e_test[i] == 1, t_test[i]) for i in range(len(e_test))],
                   dtype=[('e', bool), ('t', float)])

out_risk = model.predict_risk(x_test, times_ddh.tolist())
out_survival = model.predict_survival(x_test, times_ddh.tolist())

cis, brs, roc_auc = evaluate(model, et_train, et_test, e_test, t_test, t_train, out_risk, out_survival, times, horizons)

times_cluster = np.quantile(t, np.linspace(0, 1, 100))
clusters = model.survival_cluster(minmax(times_cluster).tolist(), 1)
plot_clusters(times_cluster, clusters)
