# %%
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Callable, TypeVar, Generic, Dict
from abc import abstractmethod, ABC
import copy

from dataclasses import dataclass
from functools import cached_property
torch.manual_seed(7)

# %% [markdown]
# # Kernel & GP Class Definitions

# %% [markdown]
# ## Abstract Kernel & Kernel Utils

# %%
class Kernel(ABC):
    @abstractmethod
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        raise NotImplementedError

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert len(x1.shape) == 2
        assert len(x2.shape) == 2
        K = self.similarity(x1[:,None,:], x2[None,:,:])
        return K

    def __add__(self, other: "Kernel") -> "Kernel":
        return SumKernel(self, other)
    
    def __mul__(self, other: "Kernel") -> "Kernel":
        return ProductKernel(self, other)
    
    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """Return a dictionary of kernel parameters."""
        raise NotImplementedError
    
class SumKernel(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return self.kernel1.similarity(x1, x2) + self.kernel2.similarity(x1, x2)

class ProductKernel(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return self.kernel1.similarity(x1, x2) * self.kernel2.similarity(x1, x2)

# %% [markdown]
# ## Kernel Implementation
# Radial Basis Function Kernel  
# Linear Kernel  
# Polynomial Kernel  
# Periodic Kernel (ExpSineSquared)

# %%
class RadialBasisFunctionKernel(Kernel):
    def __init__(self, sigma: float, length_scale: float = 1):
        self.sigma = sigma
        self.length_scale = length_scale

    def get_params(self) -> Dict[str, float]:
        return {"rbf_sigma": self.sigma, "rbf_length_scale": self.length_scale}
    
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return self.sigma**2 * torch.exp(-torch.linalg.norm(x1 - x2, dim=-1)**2 / (2 * self.length_scale**2))

class LinearKernel(Kernel):
    def __init__(self, constant: float, sigma_v: float = 1, sigma_b: float = 1):
        self.constant = constant
        self.sigma_v = sigma_v
        self.sigma_b = sigma_b

    def get_params(self) -> Dict[str, float]:
        return {"constant": self.constant, "sigma_v": self.sigma_v, "sigma_b": self.sigma_b}

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        prod = torch.einsum('ijk,ilk->ijl', x1 - self.constant, x2 - self.constant)
        return self.sigma_b**2 + (self.sigma_v**2) * prod.squeeze()

class PolynomialKernel(Kernel):
    def __init__(self, degree: int, constant: float = 1.0):
        self.degree = degree
        self.constant = constant

    def get_params(self) -> Dict[str, float]:
        return {"degree": self.degree, "constant": self.constant}
    
    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        prod = (self.constant + torch.einsum('ijk,ilk->ijl', x1, x2))**self.degree
        return prod.squeeze()

class PeriodicKernel(Kernel):
    def __init__(self, sigma: float, period: float, length_scale: float = 1):
        self.sigma = sigma
        self.period = period
        self.length_scale = length_scale
    
    def get_params(self) -> Dict[str, float]:
        return {"periodic_sigma": self.sigma, "periodic_period": self.period, "periodic_length_scale": self.length_scale}

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        return (self.sigma**2) * torch.exp(
            -2 * torch.sin(
                torch.pi * torch.linalg.vector_norm(x1 - x2, dim=-1) / self.period
            )**2 / self.length_scale**2
        )

# %% [markdown]
# ## Gaussian Process Implementation

# %%
@dataclass
class GaussianProcess:
    x : torch.Tensor # data to condition on (x coord)
    y : torch.Tensor # data to condition on (y coord)
    nx : torch.Tensor # generalization target domain; where do we want to extrapolate to?
    kernel: Kernel # defines the covariance of outputs for two inputs
    noise_sd : float = 0.1

    def __post_init__(self):

        self.all_x = self.nx
        K_star_star = self.kernel(self.nx, self.nx)

        if self.x.numel() == 0:
            self.mean = torch.zeros(len(self.all_x))
            self.cov = K_star_star
        else: #if training points are pased in, write the mean and covariance matrix
            self.all_x = torch.unique(torch.cat((self.x, self.nx), dim=0))
            self.K = self.kernel(self.x, self.x)
            K_star = self.kernel(self.x, self.nx)
            temp_K = self.K + self.noise_sd**2 * torch.eye(self.K.shape[0])
            
            self.mean = K_star.T @ torch.linalg.solve(temp_K, self.y)
            self.cov = K_star_star - K_star.T @ torch.linalg.solve(temp_K, K_star) #distance of gen pt to self
        
        self.x_order = torch.argsort(self.all_x)
    
    def likelihood(self):
        temp_cov = self.K + self.noise_sd**2 * torch.eye(self.K.shape[0]) + 1e-5 * torch.eye(self.K.shape[0])#covariance_matrix = self.K + self.noise_sd**2 * torch.eye(self.K.shape[0])
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.mean(self.y).repeat(len(self.x)),
            covariance_matrix = temp_cov
        )

        return torch.exp(mvn.log_prob(self.y))
    
    def log_likelihood(self):
        temp_cov = self.K + self.noise_sd**2 * torch.eye(self.K.shape[0]) + 1e-5 * torch.eye(self.K.shape[0])#covariance_matrix = self.K + self.noise_sd**2 * torch.eye(self.K.shape[0])
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.mean(self.y).repeat(len(self.x)),
            covariance_matrix = temp_cov
        )
        return mvn.log_prob(self.y)
    
    def sample(self) -> torch.Tensor:
        temp_cov = self.cov + 1e-9 * torch.eye(self.cov.shape[0]) #add a small amount of noise to the covariance matrix to make it positive definite
        ny = torch.distributions.MultivariateNormal(self.mean, temp_cov).sample()
        all_y = torch.cat((self.y, ny), dim=0)
        return self.all_x[self.x_order], all_y[self.x_order]

    def update(self, x: torch.Tensor = torch.tensor([])[:,None], y: torch.Tensor = torch.tensor([]), nx: torch.Tensor = torch.tensor([])[:,None]) -> "GaussianProcess":
        return GaussianProcess(
            x=torch.cat((self.x, x),dim=0),
            y=torch.cat((self.y, y),dim=0),
            nx=torch.unique(torch.cat((self.nx, nx),dim=0))[:,None],
            kernel=self.kernel,
            noise_sd=self.noise_sd
        )
    
    def plot(self, true_x=None, true_y=None, xlim=(0,6), ylim=(0,3.5)):
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
            
        nx_order = torch.argsort(self.nx)
        ax.plot(self.nx[nx_order], self.mean[nx_order], lw=3)
        err = 1.96* torch.sqrt(torch.diag(self.cov))
        ax.fill_between(
            self.nx[nx_order],
            self.mean[nx_order] - err[nx_order],
            self.mean[nx_order] + err[nx_order],
            alpha=0.3
        )
        
        if true_x is not None and true_y is not None:
            ax.plot(true_x, true_y, lw=3, color='black',linestyle = "--") #true underlying function
            
        ax.scatter(self.x, self.y, s=100, lw=3, color="red", edgecolors="black", linewidths=1,zorder=5)

        st.pyplot(fig=fig)
        # plt.show()
    
    def diag(self):
        return torch.diag(self.cov)

# # %% [markdown]
# # ### Testing GP Implementation

# # %%
# x = torch.arange(0,6,.02)
# y = x * torch.sin(x)

# # %%
# training_indices = torch.randperm(y.size(0))[:5]
# x_train, y_train = x[training_indices], y[training_indices]

# # %%
# k = RadialBasisFunctionKernel(sigma=.1, length_scale=1.5)
# gp = GaussianProcess(x=None,y=None,nx=x,kernel=k,noise_sd=.1)
# # gp.plot(ylim=(-1,1)) # RBF Prior

# # %%
# k = RadialBasisFunctionKernel(sigma=1, length_scale=1.42)
# gp = GaussianProcess(x=x_train,y=y_train,nx=x,kernel=k,noise_sd=.1)
# # gp.plot(xlim=(0,10),ylim=(-6,8))

# # %%
# gp1 = gp.update(nx=torch.arange(0,10,.02))

# # %%
# # gp1.plot(true_x=gp1.nx,true_y=gp1.nx * torch.sin(gp1.nx),xlim=(0,10),ylim=(-6,8))

# # %%
# x = torch.arange(0,10,.02)
# y = x * torch.sin(x)

# training_indices = torch.randperm(y.size(0))[:5]
# x_train, y_train = x[training_indices], y[training_indices]
# gp2 = gp1.update(x_train,y_train)
# gp2.plot(true_x=x,true_y=y,xlim=(0,10),ylim=(-6,8))

# %% [markdown]
# ## Gaussian Process Mixture Implementation

# %%
class GaussianProcessMixture:
    def __init__(self, gps: list[GaussianProcess], weights: torch.Tensor = torch.tensor([])):
        self.gps = gps
        self.num_gps = len(self.gps)
        if weights.numel() == 0:  # if weights are not provided, use uniform distribution
            self.weights = torch.full((self.num_gps,), 1/self.num_gps)
        else:
            self.weights = weights

    def sample(self):
        sampled_index = torch.multinomial(self.weights, 1).item()
        return self.gps[sampled_index].sample()
        #returns a single fn from a GP selected by weight.

    def mean(self):
        domain_means = torch.stack([gp.mean for gp in self.gps])
        mixture_means = torch.sum(domain_means * self.weights[:, None], dim=0)
        return mixture_means

    def variance(self):
        weighted_variances = torch.zeros_like(self.gps[0].mean)
        weighted_sq_means = torch.zeros_like(self.gps[0].mean)

        for i, gp in enumerate(self.gps):
            weighted_variances = weighted_variances + self.weights[i] * torch.diag(gp.cov)
            weighted_sq_means = weighted_sq_means + self.weights[i] * (gp.mean ** 2)

        weighted_means = self.mean()
        total_variance = weighted_variances + weighted_sq_means - (weighted_means ** 2)
        return total_variance
    
    def log_likelihood(self):
        log_likelihoods = torch.stack([gp.log_likelihood() for gp in self.gps])
        return torch.sum(log_likelihoods * self.weights)
    
    def highest_uncertainty_idx(self):
        domain_vars = torch.stack([gp.diag() for gp in self.gps])
        mixture_vars = torch.sum(domain_vars * self.weights[:, None], dim=0)
        return torch.argmax(mixture_vars)
    
    def get_kernels(self):
        return [gp.kernel for gp in self.gps]
        
    def update_in_place(self, x:torch.Tensor = torch.tensor([]), y:torch.Tensor = torch.tensor([]), nx:torch.Tensor = torch.tensor([])[:,None]):
        new_gps = [0] * self.num_gps
        for i, gp in enumerate(self.gps):
            new_gps[i] = gp.update(x,y,nx)
        self.gps = new_gps

    def update_copy(self, x:torch.Tensor = torch.tensor([]), y:torch.Tensor = torch.tensor([]), nx:torch.Tensor = torch.tensor([])[:,None]):
        new_gps = [gp.update(x,y,nx) for gp in self.gps]
        return new_gps

    def plot(self, num_samples, xlim=(0,6), ylim=(0,3.5)):
        for _ in range(num_samples):
            sampled_index = torch.multinomial(self.weights, 1).item()
            x, y = self.gps[sampled_index].sample()
            #plt.plot(x, y, lw=self.weights[sampled_index]*20,alpha=0.1,color='black')
            
        # plt.plot(all_x, all_y, lw=all_weights)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(-1, 4)
        plt.title('Samples from Gaussian Process Mixture')
        st.pyplot()

# %% [markdown]
# ### Testing GPM Implementation

# %%
# k1 = RadialBasisFunctionKernel(sigma=1.2, length_scale=1.4)
# k2 = RadialBasisFunctionKernel(sigma=1.2, length_scale=2)
# domain = torch.arange(0,10,.2)
# gp1 = GaussianProcess(x=torch.tensor([]),y=torch.tensor([]),nx=domain, kernel=k1)
# gp2 = GaussianProcess(x=torch.tensor([]),y=torch.tensor([]),nx=domain, kernel=k2)

# gpm = GaussianProcessMixture(gps=[gp1,gp2])

# # %%
# gpm.update_in_place(x=x_train,y=y_train)

# %%
# [gp.plot(xlim=(0,10),ylim=(-5,8)) for gp in gpm.gps]

# %% [markdown]
# # Learner and Teacher Class Definitions

# %% [markdown]
# ## Learner Implementation

# %%
class Learner:
    def __init__(self, prior_over_fams: dict, prior_over_params: dict, num_gps: int = 100, nx: torch.Tensor = torch.arange(0,6,.02), gp_distrib: GaussianProcessMixture = None):
        self.prior_over_fams = prior_over_fams
        self.prior_over_params = prior_over_params
        self.num_gps = num_gps
        self.nx = nx
        if gp_distrib is None:
            self.prior_kernels = self.generate_prior_kernels()
            self.gp_distrib = self.generate_prior_gps()
        else:
            self.gp_distrib = gp_distrib

    def sample_param_priors(self, p):
        return np.random.uniform(self.prior_over_params[p][0],self.prior_over_params[p][1])

    def generate_prior_kernels(self):
        
        kernels = [0] * self.num_gps
        k_keys = list(self.prior_over_fams.keys())
        k_weights = list(self.prior_over_fams.values())
        
        for i in range(self.num_gps):
            k_type = np.random.choice(k_keys, p=k_weights)
            if k_type == "per":
                param_names = ['sigma', 'period', 'length_scale'] 
                params = [self.sample_param_priors(p) for p in param_names]
                kernel = PeriodicKernel(*params)
            elif k_type == "poly":
                param_names = ['degree', 'constant']
                params = [self.sample_param_priors(p) for p in param_names]
                kernel = PolynomialKernel(*params)
            elif k_type == "lin":
                param_names = ['constant', 'sigma_b', 'sigma_v']
                params = [self.sample_param_priors(p) for p in param_names]
                kernel = LinearKernel(*params)
            elif k_type == "RBF":
                param_names = ['sigma', 'length_scale']
                params = [self.sample_param_priors(p) for p in param_names]
                kernel = RadialBasisFunctionKernel(*params)
            else:
                return NotImplementedError("Kernel type not implemented; please ensure type is per, poly, lin, or RBF")
            kernels[i] = kernel

        return kernels
    
    def generate_prior_gps(self):
        gps = [0] * self.num_gps
        for i in range(self.num_gps):
            kernel = self.prior_kernels[i]
            gp_noise = self.sample_param_priors('noise_sd')
            gp = GaussianProcess(x=torch.tensor([])[:,None], y=torch.tensor([]), nx=self.nx[:,None], kernel=kernel, noise_sd=gp_noise)
            gps[i] = gp
        
        return GaussianProcessMixture(gps=gps)

    def update(self,x:torch.Tensor = torch.tensor([])[:,None], y:torch.Tensor = torch.tensor([]), nx:torch.Tensor = torch.tensor([])[:,None]):
        weights = self.gp_distrib.weights
        new_gps = self.gp_distrib.update_copy(x,y,nx)
        new_gp_distrib = GaussianProcessMixture(new_gps, weights)
        #return self.gp_distrib
        return Learner(
            prior_over_fams=self.prior_over_fams,
            prior_over_params=self.prior_over_params,
            num_gps=self.num_gps,
            gp_distrib=new_gp_distrib
        )
    
    def log_likelihood(self):
        return self.gp_distrib.log_likelihood()
    
    def log_prior(self):
        kernels = self.gp_distrib.get_kernels()
        log_prior_value = 0
        for k in kernels:
            params = k.get_params()
            for key, value in params.items():
                if "sigma" in key:
                    sigma_prior = torch.distributions.Normal(1,.31)
                    log_prior_value += sigma_prior.log_prob(value) / .31
                elif "length_scale" in key:
                    length_scale_prior = torch.distributions.Normal(3,1.2)
                    # length_scale_prior = torch.distributions.Exponential()
                    log_prior_value += length_scale_prior.log_prob(value) / 1.2 #torch.log(torch.distributions.HalfNormal(1.0).log_prob(value))
                elif "period" in key:
                    period_prior = torch.distributions.Exponential(1.0) #torch.distributions.Normal(3.5,1.5)
                    log_prior_value += period_prior.log_prob(value) / 1.5 #torch.log(torch.distributions.HalfNormal(1.0).log_prob(value)) #truncated normal or normal with appropriate sd
        return log_prior_value
    
    def log_posterior(self):
        return self.log_likelihood() + self.log_prior()

    def kernel_likelihood_weighting(self, x:torch.Tensor = torch.tensor([])[:,None], y:torch.Tensor = torch.tensor([]), nx:torch.Tensor = torch.tensor([])[:,None], threshold=1e-9):
        log_weights = torch.zeros(self.num_gps)
        new_gps = self.gp_distrib.update_copy(x,y,nx) #right now, each GP is updated. could do only updating sampled ones to save time?

        for i, updated_gp in enumerate(new_gps):
            log_likelihood = updated_gp.log_likelihood()
            log_weights[i] = log_likelihood

        max_log_weight = torch.max(log_weights)
        exp_sum = torch.sum(torch.exp(log_weights - max_log_weight))
        log_Z = max_log_weight + torch.log(exp_sum)

        lw = log_weights - log_Z
        weights = torch.exp(lw)

        new_gp_distrib = GaussianProcessMixture(new_gps, weights)

        return Learner(
            prior_over_fams=self.prior_over_fams,
            prior_over_params=self.prior_over_params,
            num_gps=new_gp_distrib.num_gps,
            gp_distrib=new_gp_distrib
        )

# %% [markdown]
# ### Testing Learner Implementation

# # %%
# # start with a basic learner that doesn't have mixture and doesn't have param variance
# rbf_only = {
#     'RBF':1
# }

# basic_rbf_params = {
#     'sigma': (1,1), #per, rbf
#     'length_scale': (1.5,1.5), #per, rbf
#     'noise_sd': (1,1)
# }

# basic_learner = Learner(prior_over_fams=rbf_only, prior_over_params=basic_rbf_params, num_gps=1)

# # %%
# yo = basic_learner.gp_distrib.gps[0]
# # yo.plot(ylim=(-3,3))

# yo2 = yo.update(x=torch.tensor([2,5,8]), y= torch.tensor([-1,2,-1]))
# # yo2.plot(xlim=(-1,11),ylim=(-4,4))

# %% [markdown]
# ## Teacher Implementation

# %%
class Teacher:
    def __init__(self, true_x: torch.Tensor, true_y: torch.Tensor, learner: Learner):
        self.nx = true_x
        self.ny = true_y
        self.student = learner

    def update_learner_in_place(self, teach_x, teach_y):
        return self.student.kernel_likelihood_weighting(teach_x,teach_y,self.nx)

    def update_learner_copy(self, teach_x, teach_y):
        # return copy.deepcopy(self.student).update(teach_x,teach_y,self.nx)
        return copy.deepcopy(self.student).kernel_likelihood_weighting(teach_x,teach_y,self.nx)

    def sample_utility(self, teach_x, teach_y, num_samples):
        tempstudent = self.update_learner_copy(teach_x, teach_y)
        y_targ = self.ny
        gp_distrib = tempstudent.gp_distrib
        y_inferred_samples = torch.tensor([gp_distrib.sample() for _ in range(num_samples)])
        sampled_utility = torch.neg(torch.mean((y_inferred_samples - y_targ)**2))
        return sampled_utility, tempstudent
    
    def analytic_utility(self, teach_x, teach_y):
        tempstudent = self.update_learner_copy(teach_x, teach_y)
        y_targ = self.ny
        gp_distrib = tempstudent.gp_distrib
        y_inferred_mean = gp_distrib.mean()
        y_inferred_var = gp_distrib.variance()
        sq_errors = y_inferred_var + y_inferred_mean**2 - 2*(y_targ)*y_inferred_mean + y_targ**2
        nmse = torch.neg(torch.mean(sq_errors))
        return nmse, tempstudent
    
    def generate_candidate_points(self, num_points, num_candidate_sets):
        candidate_pt_ind = []
        candidate_pt_utils = []
        for _ in range(num_candidate_sets):
            inds = np.random.choice(np.arange(0, len(self.nx)), size=num_points, replace=False) #randomly sample 5 indices from the true function
            candidate_pt_ind.append(inds) #store the indices of the points
            xs = self.nx[inds]
            ys = self.ny[inds]
            util, tempstudent = self.analytic_utility(xs,ys) #calculate the utility of the point set
            candidate_pt_utils.append(util)
            hypothetical_student = tempstudent.gp_distrib.gps[0]
            # hypothetical_student.plot(self.nx,self.ny,xlim=(0,10),ylim=(-6,8))
        
        max_util_ind = np.argmax(candidate_pt_utils)
        return candidate_pt_ind[max_util_ind], candidate_pt_ind
    
    def generate_candidate_pointss_uncert_heuristic(self, num_points):
        teaching_pts = []
        #
        return 

# %% [markdown]
# ### Testing Teacher Implementation

# # %%
# x = torch.arange(0,10,.2)
# y = x * torch.sin(x)

# basic_learner = Learner(prior_over_fams=rbf_only, prior_over_params=basic_rbf_params, num_gps=1)
# basic_teacher = Teacher(true_x=x, true_y=y, learner=basic_learner)

# # %%
# #basic_learner.gp_distrib.gps[0].plot()
# best_set, all_sets = basic_teacher.generate_candidate_points(5,10)

# # %%
# all_sets

# # %%
# best_set

# # %%
# updated_learner = basic_teacher.update_learner_copy(teach_x=x[best_set],teach_y=y[best_set])

# %%
# updated_learner.gp_distrib.plot(num_samples=1)

# %%
# updated_learner.gp_distrib.gps[0].plot(xlim=(-1,11), ylim=(-6,6))