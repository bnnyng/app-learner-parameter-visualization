import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json

import model_torch as model
import model_functions as M
import app_functions as F

torch.manual_seed(42)

# st.title("Visualizing learner parameters")

#####   SESSION STATE VARIABLES   #####

with open('test_stim.json') as json_file:
    true_function_data = json.load(json_file)
st.session_state.sin_data = true_function_data['sin']
st.session_state.closed_form_strings = [
    f'y = {A:<.2f} * sin({B:<.2f} * x - {C:<.2f}) + {D:<.2f}'
    for i in range(len(st.session_state.sin_data))
    for (A, B, C, D) in [st.session_state.sin_data[i]['params']]
]
st.session_state.true_x = torch.arange(0, 6, 0.02)
st.session_state.train_idx = np.arange(0, 300)
st.session_state.chosen_fn_idx = None
st.session_state.optim_learners = {}

#####   SIDEBAR   #####

with st.sidebar:
    # Changes to sidebar inputs will re-run the entire script
    # Choose functions
    with st.form("sidebar"):
        chosen_function_strings = st.multiselect(
            "Select functions to visualize:",
            [f"Function {i}: {st.session_state.closed_form_strings[i]}" for i in range(len(st.session_state.sin_data))]
        )
        chosen_function_indices = [int(fn_str.split()[1][0]) for fn_str in chosen_function_strings]
        train_idx_string = st.text_input(
            "(Optional) Enter indices of training points as a list of comma-separated integers:",
            value=None
        )
        if st.form_submit_button("Select"):
            st.session_state.chosen_fn_idx = sorted(chosen_function_indices)
            try:
                train_idx_input = train_idx_string.split(',')
                train_idx_input = [int(n) for n in train_idx_input]
                st.session_state.train_idx = train_idx_input
            except:
                if train_idx_string:
                    st.write("Oops! Invalid input format.")


#####   APP BODY   #####

# Get optimal parameters for selected functions
def get_optimal_learners(n_iter=600):
    for i in st.session_state.chosen_fn_idx:
        learner = F.get_optimal_learner(
            function_i=st.session_state.sin_data[i],
            _true_x=st.session_state.true_x,
            train_idx=st.session_state.train_idx,
            n_iter=n_iter
        )
        st.session_state.optim_learners[i] = learner

@st.fragment
def display_posterior_function(fn_idx, function, params):
    col1, col2 = st.columns(2)
    updated_params = {}
    with col1:
        for k, v in params.items():
            # Ensure slider value is synced with the parameter
            slider_key = f"{fn_idx}_{k}"
            if slider_key not in st.session_state.slider_keys:
                st.session_state.slider_keys[slider_key] = v.item() if isinstance(v, torch.Tensor) else v
            # Update slider and parameter
            updated_value = st.slider(
                label=k,
                min_value=0.01,
                max_value=10.0,
                value=st.session_state.slider_keys[slider_key],  # Use session state for persistence
                key=slider_key
            )
            # Save the updated value back to the parameter dict
            updated_params[k] = torch.Tensor([updated_value])
    with col2:
        loglike, logprior, logposterior = M.evaluate_params(
            function_i=function,
            true_x=st.session_state.true_x,
            train_idx=st.session_state.train_idx,
            params=updated_params,
            plot_title=f"Closed form: {st.session_state.closed_form_strings[fn_idx]}"
        )

@st.fragment
def reset_posterior_function(fn_idx):
    function = st.session_state.sin_data[fn_idx]
    true_period = F.compute_true_period(function)
    optim_learner = st.session_state.optim_learners[fn_idx]
    optim_params = F.get_learner_params(optim_learner)

    st.markdown(f"""#### Function {fn_idx}""")
    reset_button = st.button(f"Reset", key=fn_idx)
    if reset_button or (fn_idx not in st.session_state.param_flags):
        # Reset all parameters and slider values
        st.session_state.param_flags[fn_idx] = True
        st.session_state.curr_input_params[fn_idx] = optim_params
        for k, v in optim_params.items():
            st.session_state[f"{fn_idx}_{k}"] = v.item() if isinstance(v, torch.Tensor) else v
    
    if st.session_state.param_flags[fn_idx]:
        st.session_state.param_flags[fn_idx] = False
        old_params = st.session_state.curr_input_params[fn_idx]
        st.session_state.curr_input_params[fn_idx] = display_posterior_function(fn_idx, function, old_params)


# Test parameters
st.header("Visualizing learner parameters")
if st.session_state.chosen_fn_idx is not None:
    st.session_state.param_flags = {}
    st.session_state.curr_input_params = {}
    st.session_state.slider_keys = {}
    get_optimal_learners()
    # for i, learner in st.session_state.optim_learners.items():
    #     st.write(i, F.get_learner_params(learner))
    for i in st.session_state.chosen_fn_idx:
        reset_posterior_function(i)
else:
    st.write("Use the sidebar to select functions to visualize.")


# TO DO: selecting points loss