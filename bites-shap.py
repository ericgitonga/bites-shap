import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from analyse_utils import *
from RGBSG_utils import load_RGBSG, load_RGBSG_no_onehot
import pickle
import streamlit as st
from streamlit_shap import st_shap

method = "BITES"
results_dir="example_results/"
compare_against_ATE = False

X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(partition='train', filename_="data/rgbsg.h5")
X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(partition='test', filename_="data/rgbsg.h5")

result_path=results_dir + method + "_RGBSG"

if method == "BITES":
    model, config = get_best_model(result_path)
    model.compute_baseline_hazards(X_train, [Y_train, event_train, treatment_train])
    
    C_index, C_index_T0, C_index_T1 = get_C_Index_BITES(model, X_test, Y_test, event_test, treatment_test)
    pred_ite, _ = get_ITE_BITES(model, X_test, treatment_test)
    
model.eval()

def net_treatment0(input):
    ohc = OneHotEncoder(sparse=False)
    X_ohc = ohc.fit_transform(input[:, -2:])
    tmp=np.c_[input[:, :-2], X_ohc].astype("float32")
    return model.risk_nets[0](model.shared_net(torch.tensor(tmp))).detach().numpy()

def net_treatment1(input):
    ohc = OneHotEncoder(sparse=False)
    X_ohc = ohc.fit_transform(input[:, -2:])
    tmp=np.c_[input[:, :-2], X_ohc].astype("float32")
    return model.risk_nets[1](model.shared_net(torch.tensor(tmp))).detach().numpy()

X_train, Y_train, event_train, treatment_train, _ = load_RGBSG_no_onehot(partition="train",
                                                                         filename_="data/rgbsg.h5")
X_test, Y_test, event_test, treatment_test, _ = load_RGBSG_no_onehot(partition="test",
                                                                     filename_="data/rgbsg.h5")

X_train0 = X_train[treatment_train == 0]
X_train1 = X_train[treatment_train == 1]
names = ["N pos nodes", "Age", "Progesterone", "Estrogene", "Menopause", "Grade"]
X_test0 = pd.DataFrame(X_test[treatment_test == 0], columns=names)
X_test1 = pd.DataFrame(X_test[treatment_test == 1], columns=names)

explainer_treatment0 = shap.Explainer(net_treatment0, X_train0)
explainer_treatment1 = shap.Explainer(net_treatment1, X_train1)

shap_values0_temp = explainer_treatment0(X_test0.astype("float32"))
shap_values1_temp = explainer_treatment1(X_test1.astype("float32"))

st_shap(shap.plots.beeswarm(shap_values0_temp))
st_shap(shap.plots.beeswarm(shap_values1_temp))