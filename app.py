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

# Choose training indices
true_x = torch.arange(0, 6, 0.02)
train_idx_input = None
train_idx_string = st.text_input(
    "Enter training indices as a list of comma-separated integers:",
    value=None
)
try:
    train_idx_input = train_idx_string.split(',')
    train_idx_input = [int(n) for n in train_idx_input]
except:
    st.write("Oops! Invalid input format.")
    train_idx_input = None
train_idxs = np.arange(0, 300) if train_idx_input is None else train_idx_input

# Choose functions
chosen_function_strings = st.multiselect(
    "Choose functions to visualize",
    [f"Function {i}: {closed_form_strings[i]}" for i in range(len(sin_data))]
)
chosen_function_indices = [int(fn_str.split()[1][0]) for fn_str in chosen_function_strings]

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
                max_value=float(9),
                value=v
            ) 
        input_params[fn_i] = curr_params
    return input_params

initial_optim_params = get_initial_optim_params(chosen_function_indices, n_iter=200)

###  DISPLAY MODULES  ###

def display_function_module(fn, initial_fn_params, plot_title=''):
    col1, col2 = st.columns(2)
    input_params = {}
    with col1:
        for k, v in initial_fn_params.items():
            input_params[k] = st.slider(label=k, min_value=0.01, max_value=float(9), value=v)
    with col2:
        F.evaluate_params(fn, true_x, train_idxs, input_params, plot_title=plot_title)

for i in chosen_function_indices:
    true_period = 2 * math.pi / abs(sin_data[i]['params'][1])
    display_function_module(
        fn=sin_data[i],
        initial_fn_params=initial_optim_params[i],
        plot_title=f"Function {i} with true period {true_period:<.2f}"
    )









# Get optimized parameters



# period = st.slider(label="Period parameter", min_value=0.5, max_value=float(9))



###  VISUALIZE ALL FUNCTIONS WITH VARIABLE N POINTS  ###

noise_x = np.arange(5, 100, 15)

def visualize_function_periods(function_data):
    function_y = function_data['all_y']
    true_period = 2 * math.pi / abs(function_data['params'][1])
    st.text(f"True period: {true_period}")
    with st.container():
        n_cols = len(noise_x) // 2
        columns = st.columns(n_cols)
        for i, num_known_points in enumerate(noise_x):
            known_points = np.random.choice(np.arange(300), num_known_points, replace=False)
            col_n = i % n_cols
            with columns[col_n]:
                F.evaluate_periodicity(true_x, function_y, known_points, [period], plot=True)

def visualize_range_n_points():
    with st.container(height=500):
        for fn in sin_data:
            visualize_function_periods(fn)

###  VISUALIZE ALL FUNCTIONS WITH SINGLE N POINTS  ### 

def visualize_fixed_n_points(fn_list):
    num_noise = st.slider(label="Number of noisy points", min_value=5, max_value=100, step=1)
    n_cols = len(fn_list) if len(fn_list) < 3 else 3
    columns = st.columns(n_cols)
    for i, fn in enumerate(fn_list):
        fn_y = fn['all_y']
        true_period = 2 * math.pi / abs(fn['params'][1])
        col_n = i % n_cols
        known_points = np.random.choice(np.arange(300), num_noise, replace=False)
        with columns[col_n]:
            F.evaluate_periodicity(true_x, fn_y, known_points, [period])
            st.caption(f"True period: {true_period:.4f}")

# visualize_fixed_n_points(sin_data)

###  CHOOSE FUNCTION TO VISUALIZE  ###

# chosen_fns = st.multiselect(
#     "Choose functions to visualize",
#     [f"Function {i}, true period {true_periods[i]}" for i in range(len(sin_data))]
# )
# visualize_fixed_n_points([sin_data[i] for i in range(len(chosen_fns))])