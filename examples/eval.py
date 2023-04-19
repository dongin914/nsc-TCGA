import numpy as np
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_nsc_models(best_model, x_test, t_test, e_test, t_train, e_train, times_ddh):
    out_risk = best_model.predict_risk(x_test, times_ddh.tolist())
    out_survival = best_model.predict_survival(x_test, times_ddh.tolist())

    et_train = np.array([(e_train[i] == 1, t_train[i]) for i in range(len(e_train))],
                        dtype=[('e', bool), ('t', float)])
    et_test = np.array([(e_test[i] == 1, t_test[i]) for i in range(len(e_test))],
                        dtype=[('e', bool), ('t', float)])
    selection = (t_test < t_train.max()) | (e_test == 0)

    cis = []
    for i, _ in enumerate(times_ddh):
        cis.append(concordance_index_ipcw(et_train, et_test[selection], out_risk[:, i][selection], times_ddh[i])[0])

    brs = brier_score(et_train, et_test[selection], out_survival[selection], times_ddh)[1]

    roc_auc = []
    for i, _ in enumerate(times_ddh):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test[selection], out_risk[:, i][selection], times_ddh[i])[0])

    for horizon in enumerate(times_ddh):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index:", cis[horizon[0]])
        print("Brier Score:", brs[horizon[0]])
        print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

def plot_survival_clusters(best_model, t):
    times_cluster = np.quantile(t, np.linspace(0, 1, 100))
    clusters = best_model.survival_cluster(minmax(times_cluster).tolist(), 1)

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
    plt.show()
    plt.savefig('./results.png')
    plt.close()
