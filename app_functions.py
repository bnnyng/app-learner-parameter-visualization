import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json

import model_torch as model
import model_functions as M

def compute_true_period(function):
    return 2 * math.pi / abs(function['params'][1])

@st.cache_resource
def get_optimal_learner(function_i, _true_x, train_idx, n_iter=600):
    """
    Optimize a new learner for a given function.
    """
    # st.write(true_period)
    optim_params = {
        "sigma": 1,
        "period": compute_true_period(function_i), # TO DO: set to what?
        "length_scale": 1,
        "noise_sd": 1
    }
    optim_learner, _ = M.learner_optim(
        function_i, _true_x, 
        train_idx=train_idx, 
        params=optim_params,
        n_iter=n_iter
    )
    # Get kernel parameters
    optim_params = optim_learner.gp_distrib.get_kernels()[0].get_params()
    return optim_learner

# @st.cache_data
def get_learner_params(_learner):
    params = _learner.gp_distrib.get_kernels()[0].get_params()
    params = {
        key.replace("periodic_", ""): value 
        for key, value in params.items()
    }
    return params

