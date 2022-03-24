from typing import List
import numpy as np
from matplotlib import plot as plt


# From page 6 fig 6. description
T = 500
# Define n as larger for more precision
n = 69
# From page 10
m = max(15*n, n**2)
# Define this
NUM_THETA_ROLLOUTS = 420


# Sample policy and return list of thetas
def sample_gaussian_policy(mu: List[float], sigm) -> List[float]:
    # TODO Akhil
    pass


# Given x (which is vector) what is the probability of that occuring in the gaussian function given mu and sigma
def lookup_gaussian(mu: List[float], sigma, x) -> float:
    # TODO Akhil: Equation (2)
    pass


# Bottom of page 9 formula
# covariance(t) = (1-t/T)sigma_i + t/Tsigma_f
# Returns square matrix for covariance
def get_covariance(timestep: int, dimension: int) -> np.array:
    # dynamically create sigma_i which is diag(10^-2) of n by n where n is number of thetas
    # dynamically create sigma_f which is diag(10^-5) of n by n where n is number of thetas
    # TODO: Ross
    pass


def get_uniform_k():
    # TODO: Kobe
    pass


# This function will run the ansatz circuit and then project the result onto |k>
# This is gonna the QNode then do the state projection
def evaluate_fidelity(k, theta):
    # Call QNode from policy_gradient_vqa.py
    # Then do classical path
    pass


# Implement Equation (3)
def evaluate_objective_function(mu, sigm) -> float:
    J = 0.0
    # Outer sigma
    for _ in range(m):
        # k is a quantum state which is a random sampled state
        k = get_uniform_k()
        p_k = 1 / m
        # Inner sigma
        # "Monte Carlo part"
        inner_term = 0
        for _ in range(NUM_THETA_ROLLOUTS):
            # First term of Equation 2(0)
            theta = sample_gaussian_policy(mu, sigm)
            prob = lookup_gaussian(mu, sigm, theta)
            # Output of the ansatz circuit is the second term is Equation (3)
            fid = evaluate_fidelity(k, theta)
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


MAX_ITER_REINFORCE = 999 # TODO: what should this be
# Number of times to evaluate J at a certain step for graphing purposes
GRAPH_NUM = 555

# This is the high level algorithm which does policy gradient approach
def algorithm():
    mu = 0 # TODO: choose starting mu
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