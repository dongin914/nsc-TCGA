import os
import numpy as np
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(model, et_train, et_test, e_test, t_test, t_train, out_risk, out_survival, times, horizons):
    selection = (t_test < t_train.max()) | (e_test == 0)

    cis = []
    for i, _ in enumerate(times):
        cis.append(concordance_index_ipcw(et_train, et_test[selection], out_risk[:, i][selection], times[i])[0])
    brs = brier_score(et_train, et_test[selection], out_survival[selection], times)[1]
    roc_auc = []
    for i, _ in enumerate(times):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test[selection], out_risk[:, i][selection], times[i])[0])

    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index:", cis[horizon[0]])
        print("Brier Score:", brs[horizon[0]])
        print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

    return cis, brs, roc_auc

def plot_clusters(times_cluster, clusters):
    fig, ax = plt.subplots()

    for cluster in range(clusters.shape[1]):
        cluster_survival = clusters[:, cluster]
        step_times = np.repeat(times_cluster, 2)[1:]
        step_survival = np.repeat(cluster_survival, 2)[:-1]
        ax.plot(step_times, step_survival, label=f'Cluster {cluster}')

    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    ax.legend(title='Clusters')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    if not os.path.exists('../results'):
        os.makedirs('../results')
    plt.savefig('../results/nsc_TCGA.png')
    plt.close()
