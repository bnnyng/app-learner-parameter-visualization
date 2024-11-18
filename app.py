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

###  IMPORT FUNCTION DATA  ###

with open('test_stim.json') as json_file:
    true_function_data = json.load(json_file)

sin_data = true_function_data['sin']
true_x = torch.arange(0, 6, 0.02)
noise_x = np.arange(5, 100, 15)

sin = sin_data[0]['all_y']

st.title("Visualizing learner parameters")


period = st.slider(label="Period parameter", min_value=0.5, max_value=float(9))
num_noise = st.slider(label="Number of noisy points", min_value=5, max_value=100, step=1)

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

# with st.container(height=500):
#     for fn in sin_data:
#         visualize_function_periods(fn)

columns = st.columns(5)
for i, fn in enumerate(sin_data):
    fn_y = fn['all_y']
    true_period = 2 * math.pi / abs(fn['params'][1])
    col_n = i % 5
    known_points = np.random.choice(np.arange(300), num_noise, replace=False)
    with columns[col_n]:
        F.evaluate_periodicity(true_x, fn_y, known_points, [period])