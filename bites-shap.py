import shap
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from analyse_utils import get_C_Index_BITES, get_best_model
from analyse_utils import get_ITE_BITES, analyse_randomized_test_set
from RGBSG_utils import load_RGBSG, load_RGBSG_no_onehot
import streamlit as st
import streamlit_analytics
import pandas as pd
import torch
import matplotlib.pyplot as plt

with streamlit_analytics.track():
    st.title("Displaying BITES SHAP in Streamlit")

    method = "BITES"
    results_dir = "example_results/"
    # compare_against_ATE = False

    X_train, Y_train, event_train, treatment_train, _, _ = load_RGBSG(
        partition='train', filename_="data/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _, _ = load_RGBSG(
        partition='test', filename_="data/rgbsg.h5")
  
    result_path = results_dir + method + "_RGBSG"
    st.markdown("### Survival Time vs. Survival Probability")
    toggle_ATE = st.checkbox("Show baseline plot", key="ATE")

    if method == "BITES":
        model, config = get_best_model(result_path)
        model.compute_baseline_hazards(
            X_train, [Y_train, event_train, treatment_train])

        C_index, C_index_T0, C_index_T1 = get_C_Index_BITES(
            model, X_test, Y_test, event_test, treatment_test)
        pred_ite, _ = get_ITE_BITES(model, X_test, treatment_test)
    
        if toggle_ATE: 
        # if compare_against_ATE:
            analyse_randomized_test_set(np.ones_like(pred_ite), Y_test,
                                        event_test, treatment_test,
                                        C_index=C_index, method_name=None,
                                        annotate=False)
            fig, ax = plt.subplots()
            ax = analyse_randomized_test_set(pred_ite, Y_test, event_test,
                                             treatment_test, C_index=C_index,
                                             method_name=method,
                                             new_figure=False, annotate=True)
        else:
             fig, ax = plt.subplots()
             ax = analyse_randomized_test_set(pred_ite, Y_test, event_test,
                                              treatment_test, C_index=C_index,
                                              method_name=method)
        with st.expander("Show Plot"):
            st.pyplot(fig)

    model.eval()

    def net_treatment0(input):
        ohc = OneHotEncoder(sparse=False)
        X_ohc = ohc.fit_transform(input[:, -2:])
        tmp = np.c_[input[:, :-2], X_ohc].astype("float32")
        return model.risk_nets[0](model.shared_net(torch.tensor(tmp))).detach().numpy()


    def net_treatment1(input):
        ohc = OneHotEncoder(sparse=False)
        X_ohc = ohc.fit_transform(input[:, -2:])
        tmp = np.c_[input[:, :-2], X_ohc].astype("float32")
        return model.risk_nets[1](model.shared_net(torch.tensor(tmp))).detach().numpy()


    X_train, Y_train, event_train, treatment_train, _ = load_RGBSG_no_onehot(partition="train",
                                                                             filename_="data/rgbsg.h5")
    X_test, Y_test, event_test, treatment_test, _ = load_RGBSG_no_onehot(partition="test",
                                                                         filename_="data/rgbsg.h5")

    X_train0 = X_train[treatment_train == 0]
    X_train1 = X_train[treatment_train == 1]
    names = ["N pos nodes", "Age", "Progesterone",
             "Estrogene", "Menopause", "Grade"]
    X_test0 = pd.DataFrame(X_test[treatment_test == 0], columns=names)
    X_test1 = pd.DataFrame(X_test[treatment_test == 1], columns=names)

    explainer_treatment0 = shap.Explainer(net_treatment0, X_train0)
    explainer_treatment1 = shap.Explainer(net_treatment1, X_train1)

    shap_values0_temp = explainer_treatment0(X_test0.astype("float32"))
    shap_values1_temp = explainer_treatment1(X_test1.astype("float32"))
    
    st.markdown("### Collective Beeswarm plots")
    with st.expander("Show Plots"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Non-Hormonal Treatment**")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values0_temp)
            st.pyplot(fig)

        with c2:           
            st.markdown("**Hormonal Treatment**")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values1_temp)
            st.pyplot(fig)

    st.markdown("### Individual Beeswarm and Waterfall plots")
    with st.expander("Show Plots"):
        index0 = st.selectbox("Select Non-Hormonal Treatment Patient To Display",
                              list(range(1, len(X_test0)+1)), index=0)
        shap_object0 = shap.Explanation(base_values=shap_values0_temp[index0-1][0].base_values,
                                        values=shap_values0_temp[index0-1].values,
                                        feature_names=X_test0.columns,
                                        data=shap_values0_temp[index0-1].data)
        st.markdown("**Non-Hormonal Treatment Patient {}**".format(index0))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Beeswarm Plot**")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values0_temp[index0-1:index0])
            st.pyplot(fig)

        with c2:
            st.markdown("**Waterfall Plot**")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_object0)
            st.pyplot(fig)
        
        index1 = st.selectbox("Select Hormonal Treatment Patient To Display",
                              list(range(1, len(X_test1)+1)), index=0)
        shap_object1 = shap.Explanation(base_values=shap_values1_temp[index1-1][0].base_values,
                                        values=shap_values1_temp[index1-1].values,
                                        feature_names=X_test1.columns,
                                        data=shap_values1_temp[index1-1].data)
        st.markdown("**Hormonal Treatment Patient {}**".format(index1))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Beeswarm Plot**")
            fig, ax = plt.subplots()
            shap.plots.beeswarm(shap_values1_temp[index1-1:index1])
            st.pyplot(fig)

        with c2:
            st.markdown("**Waterfall Plot**")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_object1)
            st.pyplot(fig)
        