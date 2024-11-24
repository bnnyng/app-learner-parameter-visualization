import torch
import numpy as np
import model_torch as model
import matplotlib.pyplot as plt
import streamlit as st
import tqdm

# Function to easily build a new GP Mixture object
def new_gpm(kernel_params: dict, true_x: list):
    gps = []

    def initialize_kernel(name, params):
        if name == "periodic":
            return model.PeriodicKernel(
                sigma=params.get("sigma", 1),
                period=params.get("period", 1),
                length_scale=params.get("length_scale", 1),
                # noise_sd=params.get("noise_sd", 1)
            )
        elif name == "linear":
            return model.LinearKernel(
                constant=params.get("constant", 1),
                sigma_v=params.get("sigma_v", 1),
                sigma_b=params.get("sigma_b", 1)
            )
        elif name == "rbf":
            return model.RadialBasisFunctionKernel(
                sigma=params.get("sigma", 1),
                length_scale=params.get("length_scale", 1)
            )
        elif name == "polynomial":
            return model.PolynomialKernel(
                degree=params.get("degree", 2),
                constant=params.get("constant", 1)
            )
        else:
            raise ValueError(f"Unknown kernel name: {name}. \n Valid names are: periodic, linear, rbf, polynomial")

    for name, params in kernel_params.items():
        kernel = initialize_kernel(name, params)
        gps.append(model.GaussianProcess(
            x=torch.tensor([])[:, None],
            y=torch.tensor([]),
            nx=true_x[:, None],
            kernel=kernel,
            noise_sd=params.get("noise_sd", 0.2)
        ))

    return model.GaussianProcessMixture(gps)

def plot_gp(learner: model.Learner, ys=None, show_known_points=True):
    # Extract known points (training data)?
    known_xs = learner.gp_distrib.gps[0].x.flatten().detach().numpy()
    known_ys = learner.gp_distrib.gps[0].y.detach().numpy()

    domain_xs = learner.nx.flatten().detach().numpy()
    test_ys = learner.gp_distrib.mean()
    # st.write(domain_xs)
    # st.write(test_ys)

    fig, ax = plt.subplots()
    if show_known_points:
        ax.plot(known_xs, known_ys, 'ro')
    ax.plot(domain_xs, test_ys.detach().numpy(), linewidth = 3)
    ax.fill_between(
        domain_xs,
        learner.gp_distrib.mean().detach().numpy() - 1.96* torch.sqrt(learner.gp_distrib.variance()).detach().numpy(),
        learner.gp_distrib.mean().detach().numpy() + 1.96* torch.sqrt(learner.gp_distrib.variance()).detach().numpy(),
        alpha=0.3
    )
    if ys != None:
        ax.plot(domain_xs, ys, linestyle='--', color='black', linewidth = 3)
        ax.set_ylim(0,3.5)
    return fig, ax

def learner_optim(function_i, true_x, train_idx, params, n_iter=600):
    # Get closed form of function
    A, B, C, D = function_i['params']
    target_fn = lambda x: A * torch.sin(B * x - C) + D

    train_xs = torch.tensor([true_x[i].item() for i in train_idx])
    train_ys = target_fn(train_xs)
    
    for k, v in params.items():
        params[k] = torch.Tensor([v]).requires_grad_(True)

    gp_k = new_gpm({"periodic": params}, train_xs)
    learner_optimizer = torch.optim.Adam(
        [v for k, v in params.items() if v.requires_grad],
        lr=0.01
    )
    learner = model.Learner(
        prior_over_fams= {},
        prior_over_params={},
        nx=true_x[:,None],
        num_gps=gp_k.num_gps,
        gp_distrib = gp_k
    )
        
    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        learner_optimizer.zero_grad()
        learner_updated = learner.update(x=train_xs[:,None], y=train_ys)

        logpost = learner_updated.log_posterior()
        (-logpost).backward()
        # iterator.set_description(
        #     f"post: {logpost.item():.2f}; pd: {params['period'].item():.2f}; n: {params['noise_sd'].item():.2f}; ls: {params['length_scale'].item():.2f}; s: {params['sigma'].item():.2f}"
        # )
        learner_optimizer.step()
        params['noise_sd'].data.clamp_(0.1, None)
        params['length_scale'].data.clamp_(0.01, None)
        params['sigma'].data.clamp_(0.01, None)
        params['period'].data.clamp_(0.01, None)
    
    return learner, train_ys

def evaluate_params(function_i, true_x, train_idx, params, plot=True, show_known_points=True, plot_title=''):
    A, B, C, D = function_i['params']
    target_fn = lambda x: A * torch.sin(B * x - C) + D

    train_xs = torch.tensor([true_x[i].item() for i in train_idx])
    train_ys = target_fn(train_xs)
    # st.write(train_xs)
    # st.write(train_ys)
    train_ys += torch.randn_like(train_ys) * .01
    # train_ys = torch.tensor([ys[i] for i in train_idx])
    
    # tensor_params = {
    #     'sigma': torch.Tensor([params['periodic_sigma']]),
    #     'period': torch.Tensor([params['periodic_period']]),
    #     'length_scale': torch.Tensor([params['periodic_length_scale']]),
    #     'noise_sd': torch.Tensor([1])
    # }

    tensor_params = {
        'sigma': params["sigma"],
        'period': params["period"],
        'length_scale': params["length_scale"],
        'noise_sd': torch.Tensor([1])
    }

    gp_k = new_gpm({"periodic": tensor_params}, true_x=true_x)

    learner = model.Learner(
        prior_over_fams={},
        prior_over_params={},
        nx=true_x[:, None],
        num_gps=gp_k.num_gps,
        gp_distrib=gp_k
    )
    learner_updated = learner.update(x=train_xs[:, None], y=train_ys)

    loglike = learner_updated.log_likelihood()
    logprior = learner_updated.log_prior()
    logposterior = loglike + logprior

    if plot:
        fig, ax = plot_gp(learner_updated, ys=target_fn(true_x), show_known_points=show_known_points)
        fig.suptitle(plot_title)
        st.pyplot(fig)
    
    return loglike, logprior, logposterior