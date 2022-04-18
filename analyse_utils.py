import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bites.model.BITES_base import BITES
from bites.utils.eval_surv import EvalSurv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from ray.tune import Analysis


def get_best_model(path_to_experiment="./ray_results/test_hydra", assign_treatment=None):
    analysis = Analysis(path_to_experiment, default_metric="val_loss", default_mode="min")
    best_config = analysis.get_best_config()
    best_checkpoint_dir = analysis.get_best_checkpoint(analysis.get_best_logdir())

    if best_config["Method"] == 'BITES' or best_config["Method"] == 'ITES':
        best_net = BITES(best_config["num_covariates"], best_config["shared_layer"], best_config["individual_layer"],
                         out_features=1,
                         dropout=best_config["dropout"])
    
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"), map_location=torch.device('cpu'))

    best_net.load_state_dict(model_state)

    return best_net, best_config


def get_C_Index_BITES(model, X, time, event, treatment):
    if not model.baseline_hazards_:
        print('Compute Baseline Hazards before running get_C_index')
        return

    surv0, surv1 = model.predict_surv_df(X, treatment)
    surv = pd.concat([surv0, surv1], axis=1)
    surv = surv.interpolate('index')
    C_index0 = EvalSurv(surv0, time[treatment == 0], event[treatment == 0], censor_surv='km').concordance_td()
    C_index1 = EvalSurv(surv1, time[treatment == 1], event[treatment == 1], censor_surv='km').concordance_td()
    C_index = EvalSurv(surv, np.append(time[treatment == 0], time[treatment == 1]),
                       np.append(event[treatment == 0], event[treatment == 1]),
                       censor_surv='km').concordance_td()

    return C_index, C_index0, C_index1


def get_ITE_BITES(model, X, treatment, best_treatment=None, death_probability=0.5):
    if not model.baseline_hazards_:
        print('Compute Baseline Hazards before running get_ITE()')
        return

    def find_nearest_index(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    surv0, surv1 = model.predict_surv_df(X, treatment)
    surv0_cf, surv1_cf = model.predict_surv_counterfactual_df(X, treatment)

    """Find factual and counterfactual prediction: Value at 50% survival probability"""
    pred0 = np.zeros(surv0.shape[1])
    pred0_cf = np.zeros(surv0.shape[1])
    for i in range(surv0.shape[1]):
        pred0[i] = surv0.axes[0][find_nearest_index(surv0.iloc[:, i].values, death_probability)]
        pred0_cf[i] = surv0_cf.axes[0][find_nearest_index(surv0_cf.iloc[:, i].values, death_probability)]
    ITE0 = pred0_cf - pred0

    pred1 = np.zeros(surv1.shape[1])
    pred1_cf = np.zeros(surv1.shape[1])
    for i in range(surv1.shape[1]):
        pred1[i] = surv1.axes[0][find_nearest_index(surv1.iloc[:, i].values, death_probability)]
        pred1_cf[i] = surv1_cf.axes[0][find_nearest_index(surv1_cf.iloc[:, i].values, death_probability)]
    ITE1 = pred1 - pred1_cf

    ITE = np.zeros(X.shape[0])
    k, j = 0, 0
    for i in range(X.shape[0]):
        if treatment[i] == 0:
            ITE[i] = ITE0[k]
            k = k + 1
        else:
            ITE[i] = ITE1[j]
            j = j + 1

    correct_predicted_probability=None
    if best_treatment is not None:
        correct_predicted_probability=np.sum(best_treatment==(ITE>0)*1)/best_treatment.shape[0]
        print('Fraction best choice: ' + str(correct_predicted_probability))

    return ITE, correct_predicted_probability


def analyse_randomized_test_set(pred_ite, Y_test, event_test, treatment_test,
                                C_index=None, method_name='set_name',
                                save_path=None, new_figure=True, annotate=True):
    mask_recommended = (pred_ite > 0) == treatment_test
    mask_antirecommended = (pred_ite < 0) == treatment_test

    recommended_times = Y_test[mask_recommended]
    recommended_event = event_test[mask_recommended]
    antirecommended_times = Y_test[mask_antirecommended]
    antirecommended_event = event_test[mask_antirecommended]

    logrank_result = logrank_test(recommended_times, antirecommended_times, recommended_event, antirecommended_event, alpha=0.95)

    colors = sns.color_palette()
    kmf = KaplanMeierFitter()
    kmf_cf = KaplanMeierFitter()
    if method_name==None:
        kmf.fit(recommended_times, recommended_event, label='Treated')
        kmf_cf.fit(antirecommended_times, antirecommended_event, label='Control')
    else:
        kmf.fit(recommended_times, recommended_event, label=method_name + ' Recommendation')
        kmf_cf.fit(antirecommended_times, antirecommended_event, label=method_name + ' Anti-Recommendation')


    if new_figure:
        if method_name==None:
            kmf.plot(c=colors[0],ci_show=False)
            kmf_cf.plot(c=colors[1],ci_show=False)
        else:
            kmf.plot(c=colors[0])
            kmf_cf.plot(c=colors[1])
    else:
        kmf.plot(c=colors[2])
        kmf_cf.plot(c=colors[3])


    if annotate:
        # Calculate p-value text position and display.
        y_pos = 0.4
        plt.text(1 * 3, y_pos, f"$p$ = {logrank_result.p_value:.6f}", fontsize='small')
        fraction2 = np.sum((pred_ite > 0)) / pred_ite.shape[0]
        plt.text(1 * 3, 0.3, 'C-Index=' + str(C_index)[:5], fontsize='small')
        plt.text(1 * 3, 0.2, f"{fraction2 * 100:.1f}% recommended for T=1", fontsize='small')

    plt.xlabel('Survival Time [month]')
    plt.ylabel('Survival Probability')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')

