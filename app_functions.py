import torch
import numpy as np
import model_torch as model
import matplotlib.pyplot as plt
import streamlit as st

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


def evaluate_periodicity(xs, ys, train_idxs, period_samples, plot=True):
    utilities = np.zeros_like(period_samples)
    logpriors = np.zeros_like(period_samples)
    logposteriors = np.zeros_like(period_samples)
    periods = []

    train_xs = torch.tensor([xs[i].item() for i in train_idxs])
    train_ys = torch.tensor([ys[i] for i in train_idxs])
    train_ys += torch.randn_like(train_ys) * .01

    max_logposterior = -float('inf')
    for i in range(len(period_samples)):
        period = period_samples[i]

        params = {
            "sigma": torch.Tensor([1]),
            "period": torch.Tensor([period]),
            "length_scale": torch.Tensor([1]),
            "noise_sd": torch.Tensor([1]),
            "true_x": torch.Tensor(xs)
            # "sigma": 1,
            # "period": period,
            # "length_scale": 1,
            # "noise_sd": .05
        }

        gp_k = new_gpm({"periodic": params}, true_x=xs)

        learner = model.Learner(
            prior_over_fams={},
            prior_over_params={},
            nx=xs[:, None],
            num_gps=gp_k.num_gps,
            gp_distrib=gp_k
        )

        learner_updated = learner.update(x=train_xs[:, None], y=train_ys)

        loglike = learner_updated.log_likelihood()
        logprior = learner_updated.log_prior()
        logposterior = loglike + logprior

        utilities[i] = loglike
        logpriors[i] = logprior
        logposteriors[i] = logposterior
        if max_logposterior < logposterior.item():
            max_logposterior = logposterior.item()
            best_learner = learner_updated

    best_period = best_learner.gp_distrib.get_kernels()[0].get_params()['periodic_period'].item()

    if plot:

        fig, ax = plot_gp(best_learner, ys, show_known_points=True)
        fig.suptitle(f'Predictions With {len(train_xs)} Noisy Points')
        st.pyplot(fig)
        # print("max logposterior", max_logposterior)
        # print("max posterior periodicity", best_learner.gp_distrib.get_kernels()[0].get_params()['periodic_period'].item())

        # fig, ax = plt.subplots()
        # ax.plot(period_samples, logposteriors)
        # fig.suptitle(f'LogPosterior vs. Periodicity Parameter w/ {len(train_xs)} Noisy Points')
        # ax.set_xlabel('periodicity parameter')
        # st.pyplot(fig)

    return utilities, logpriors, logposteriors, best_period

def plot_gp(learner: model.Learner, ys=None, show_known_points=True):
    # Extract known points (training data)?
    known_xs = learner.gp_distrib.gps[0].x.flatten().detach().numpy()
    known_ys = learner.gp_distrib.gps[0].y.detach().numpy()

    domain_xs = learner.nx.flatten().detach().numpy()
    test_ys = learner.gp_distrib.mean()
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