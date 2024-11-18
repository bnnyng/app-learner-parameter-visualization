import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json

import model_torch as model
import app_functions as F

torch.manual_seed(42)

###  INITIAL VALUES  ###

st.title("Visualizing learner parameters")

# Import function data
with open('test_stim.json') as json_file:
    true_function_data = json.load(json_file)
sin_data = true_function_data['sin']
closed_form_strings = [
    f'y = {A:<.2f} * sin({B:<.2f} * x - {C:<.2f}) + {D:<.2f}'
    for i in range(len(sin_data))
    for (A, B, C, D) in [sin_data[i]['params']]
]

# Choose functions
chosen_function_strings = st.multiselect(
    "Select functions to visualize:",
    [f"Function {i}: {closed_form_strings[i]}" for i in range(len(sin_data))]
)
chosen_function_indices = [int(fn_str.split()[1][0]) for fn_str in chosen_function_strings]

# Choose training indices
true_x = torch.arange(0, 6, 0.02)
train_idx_input = None
train_idx_string = st.text_input(
    "(Optional) Enter training indices as a list of comma-separated integers:",
    value=None
)
try:
    train_idx_input = train_idx_string.split(',')
    train_idx_input = [int(n) for n in train_idx_input]
except:
    # st.write("Oops! Invalid input format.")
    train_idx_input = None
train_idxs = np.arange(0, 300) if train_idx_input is None else train_idx_input

# Initial optimal inputs
def get_initial_learner(function_i, n_iter=600):
    true_period = 2 * math.pi / abs(function_i['params'][1])
    optim_params = {
        "sigma": 1,
        "period": true_period,
        "length_scale": 1,
        "noise_sd": 1
    }
    optim_learner, _ = F.learner_optim(
        function_i, true_x, 
        train_idxs=train_idxs, 
        params=optim_params,
        n_iter=n_iter
    )
    return optim_learner

def get_initial_optim_params(fn_indices, n_iter=600):
    optim_params = {}
    for i in fn_indices:
        tensor_params = get_initial_learner(
            sin_data[i], n_iter=n_iter
        ).gp_distrib.get_kernels()[0].get_params()
        optim_params[i] = {k: v.item() for k, v in tensor_params.items()}
    return optim_params

def get_input_params(initial_optim_params):
    input_params = {}
    for fn_i, params in initial_optim_params.items():
        curr_params = {}
        for k, v in params.items():
            curr_params[k] = st.slider(
                label=k,
                min_value=0.01,
                max_value=float(10),
                value=v
            ) 
        input_params[fn_i] = curr_params
    return input_params

initial_optim_params = get_initial_optim_params(chosen_function_indices, n_iter=200)

###  DISPLAY MODULES  ###

def display_function_module(fn, fn_i, initial_fn_params, plot_title='', closed_form=None, true_period=None):
    col1, col2 = st.columns(2)
    input_params = {}
    with col1:
        st.write(f"Parameters for function {fn_i} with true period {true_period:<.2f}:")
        for k, v in initial_fn_params.items():
            input_params[k] = st.slider(label=k, min_value=0.01, max_value=float(10), value=v)
    with col2:
        F.evaluate_params(fn, true_x, train_idxs, input_params, plot_title=plot_title)

###  APPLICATION BODY  ###

for i in chosen_function_indices:
    display_function_module(
        fn=sin_data[i],
        fn_i=i,
        initial_fn_params=initial_optim_params[i],
        plot_title=f"{closed_form_strings[i]}",
        true_period=2 * math.pi / abs(sin_data[i]['params'][1])    
    )



