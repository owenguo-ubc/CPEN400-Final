from typing import List
from constants import *
import numpy as np
from policy_gradient_vqa import *
from scipy.stats import multivariate_normal


# Sample policy and return list of thetas
def sample_gaussian_policy(mu: List[float], sigm) -> List[float]:
    return np.random.multivariate_normal(mu, sigm)


# Given x (which is vector) what is the probability of that occuring in the gaussian function given mu and sigma
def lookup_gaussian(mu: List[float], sigma, x) -> float:
    # TODO Akhil: Equation (2)
    return multivariate_normal.pdf(x, mean=mu, cov=sigma)


def get_covariance(timestep: int) -> np.array:
    """
    This implements the formula on the bottom of page 9
    covariance(t) = ((1 - t/ T) * sigma_i) + (t / (T * sigma_f))
    Returns square matrix for covariance

    :param timestep: The timestep at which to get the covariance
    :param dimension: The dimension of the diagonal matrices
    """
    # Dynamically create sigma_i which is diag(10^-2) of n by n where n is number of thetas
    sigma_i = np.diag([10**-2 for _ in range(N_VAL)])
    # Dynamically create sigma_f which is diag(10^-5) of n by n where n is number of thetas
    sigma_f = np.diag([10**-5 for _ in range(N_VAL)])
    # Compute the covariance to return
    covariance = (((1 - timestep) / T_VAL) * sigma_i) + ((timestep / T_VAL) * sigma_f)
    return covariance


def get_uniform_k():
    """Placeholder random uniform sample

    :param num_qubits: How many qubits to create a random state for

    """
    # TODO: needs actual implementation
    return np.random.uniform(low=0, high=1, size=NUM_QUBITS)


def evaluate_fidelity(unitary, thetas: List[float], k):
    """
    This function will run the ansatz circuit and then project the result onto |k>
    This is gonna the QNode then do the state projection

    :param thetas: List of thetas
    :param k: Quantum state which is a random sampled state
    """
    # Call QNode from policy_gradient_vqa.py
    qnode = build_vqa_qnode(unitary)
    state = qnode(k, thetas, NUM_LAYERS)
    # Project the resulting state from calling qnode onto |k>
    return projection_norm_squared(state, k).real


# Implement Equation (3)
def evaluate_objective_function(unitary, mu, sigm) -> float:
    J = 0.0
    # Outer sigma
    for _ in range(M_VAL):
        # k is a quantum state which is a random sampled state
        k = get_uniform_k()
        p_k = 1 / M_VAL

        theta = sample_gaussian_policy(mu, sigm)
        fid = evaluate_fidelity(unitary, theta, k)

        J += (p_k * fid)

    return J


def estimate_gradient(unitary, mu, sigma):
    """"
    Implements equation 4 found on page 3
    """
    J_delta = np.zeros(N_VAL)

    for _ in range(M_VAL):
        # k is a quantum state which is a random sampled state
        k = get_uniform_k()
        p_k = 1 / M_VAL

        theta = sample_gaussian_policy(mu, sigma)
        # Output of the ansatz circuit is the second term is Equation (3)
        fid = evaluate_fidelity(unitary, theta, k)
        log_mu_gradient_estimate = log_likelyhood_gradient_mu(mu, sigma, theta)

        J_delta += (p_k * fid * log_mu_gradient_estimate)

    return J_delta


def log_likelyhood_gradient_mu(mu, sigma, thetas):
    return np.linalg.inv(sigma).dot((thetas - mu))


def gradient_variance(previous_variance, current_gradient):
    return ( (GAMMA * previous_variance) + (1 - GAMMA) * (np.dot(current_gradient, current_gradient)) )


# This would be doing one step of optimizing mu after gradient estimation
# Needs to implement Equations (12) and (13)
def step_and_optimize_mu(old_mu, previous_variance, mu_gradient) -> tuple[List[float], float]:
    new_variance = gradient_variance(previous_variance, mu_gradient)
    new_mu = old_mu + ( ETA * (mu_gradient / np.sqrt((new_variance + EPSILON)) ) )
    return (new_mu, new_variance)


# This is the high level algorithm which does policy gradient approach
def pgrl_algorithm(unitary):
    mu = np.zeros(N_VAL)  # TODO: choose starting mu
    sigma = get_covariance(0)
    # initial variance ???
    gradient_variance = 0

    J = []
    for i in range(1, MAX_ITER_REINFORCE):

        J_step = []
        for _ in range(GRAPH_NUM):
            J_step.append(evaluate_objective_function(unitary, mu, sigma))

        J.append(J_step)

        grad_est = estimate_gradient(unitary, mu, sigma)
        mu, gradient_variance = step_and_optimize_mu(mu, gradient_variance, grad_est)
        sigma = get_covariance(i)


    # At the very end have optimized mu, sigma
    return mu, sigma, J
