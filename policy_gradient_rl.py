from typing import List
from constants import *
import numpy as np
from matplotlib import plot as plt

# Sample policy and return list of thetas
def sample_gaussian_policy(mu: List[float], sigm) -> List[float]:
    # TODO Akhil
    pass


# Given x (which is vector) what is the probability of that occuring in the gaussian function given mu and sigma
def lookup_gaussian(mu: List[float], sigma, x) -> float:
    # TODO Akhil: Equation (2)
    pass


def get_covariance(timestep: int, dimension: int) -> np.array:
    """
    This implements the formula on the bottom of page 9
    covariance(t) = ((1 - t/ T) * sigma_i) + (t / (T * sigma_f))
    Returns square matrix for covariance

    :param timestep: The timestep at which to get the covariance
    :param dimension: The dimension of the diagonal matrices
    """
    # Dynamically create sigma_i which is diag(10^-2) of n by n where n is number of thetas
    sigma_i = np.diag(np.full(N_VAL, 10**-2))
    # Dynamically create sigma_f which is diag(10^-5) of n by n where n is number of thetas
    sigma_f = np.diag(np.full(N_VAL, 10**-5))
    # Compute the covirance to return
    covariance = (((1 - timestep) / T_VAL) * sigma_i) + (timestep / (T_VAL * sigma_f))
    return covirance


def get_uniform_k():
    # TODO: Kobe
    pass


def evaluate_fidelity(thetas: List[float], k):
    """
    This function will run the ansatz circuit and then project the result onto |k>
    This is gonna the QNode then do the state projection

    :param thetas: List of thetas
    :param k: Quantum state which is a random sampled state
    """
    # Call QNode from policy_gradient_vqa.py
    qnode = build_vqa_qnode(thetas)
    state = qnode(k, thetas)
    # Project the resulting state from calling qnode onto |k>
    projection_norm_squared(state, k)


# Implement Equation (3)
def evaluate_objective_function(mu, sigm) -> float:
    J = 0.0
    # Outer sigma
    for _ in range(M_VAL):
        # k is a quantum state which is a random sampled state
        k = get_uniform_k()
        p_k = 1 / M_VAL
        # Inner sigma
        # "Monte Carlo part"
        inner_term = 0
        for _ in range(NUM_THETA_ROLLOUTS):
            # First term of Equation 2(0)
            theta = sample_gaussian_policy(mu, sigm)
            prob = lookup_gaussian(mu, sigm, theta)
            # Output of the ansatz circuit is the second term is Equation (3)
            fid = evaluate_fidelity(theta, k)
            inner_term = prob * fid
        J += p_k * inner_term
    return J


# Super High Level Realm, still need to fledge out

# This would be implementing Equation (4)
def estimate_gradient():
    pass


# This would be doing one step of optimizing mu after gradient estimation
# Needs to implement Equations (12) and (13)
def step_and_optimize_mu(old_mu) -> List[float]:
    return new_mu


# This is the high level algorithm which does policy gradient approach
def algorithm():
    mu = 0  # TODO: choose starting mu
    sigma = get_covariance(0)

    J = []
    for _ in range(MAX_ITER_REINFORCE):

        J_step = []
        for _ in range(GRAPH_NUM):
            J_step.append(evaluate_objective_function(mu, signm))

        J.append(J_step)

        # unsure right now, think we need many gradient estimates???
        grad_est = estimate_gradient()
        mu = step_and_optimize_mu(mu)

    # We plot our graph of the objective function over time
    plt.plot(J)

    # At the very end have optimized mu, sigma
    return mu, sigma