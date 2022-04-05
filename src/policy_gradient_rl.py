from typing import List
from .constants import *
import numpy as np
from .policy_gradient_vqa import *
from scipy.stats import multivariate_normal
from multiprocessing import Pool

RAND_SEED = np.random.default_rng()
NUM_ROLLOUT = 20

def _sample_gaussian_policy(mu: List[float], sigm) -> List[float]:
    """
    Sample policy and return list of thetas
    """
    return np.random.multivariate_normal(mu, sigm)


def _lookup_gaussian(mu: List[float], sigma, x) -> float:
    """
    Given x (which is vector) what is the probability of that occuring in the gaussian function given mu and sigma
    """
    return multivariate_normal.pdf(x, mean=mu, cov=sigma)


def _get_covariance(n_val: int, timestep: int) -> np.array:
    """
    This implements the formula on the bottom of page 9
    covariance(t) = ((1 - t/ T) * sigma_i) + (t / (T * sigma_f))
    Returns square matrix for covariance

    :param timestep: The timestep at which to get the covariance
    :param dimension: The dimension of the diagonal matrices
    """
    # Dynamically create sigma_i which is diag(10^-2) of n by n where n is number of thetas
    sigma_i = np.diag([10 ** -2 for _ in range(n_val)])
    # Dynamically create sigma_f which is diag(10^-5) of n by n where n is number of thetas
    sigma_f = np.diag([10 ** -5 for _ in range(n_val)])
    # Compute the covariance to return
    covariance = ((1 - (timestep / T_VAL)) * sigma_i) + ((timestep / T_VAL) * sigma_f)
    return covariance


def _get_uniform_k(num_qubits):
    """Generator a random state vector.

    Largely derived from the Qiskit implementation of random_statevector(dims, seed=None)

    The statevector is sampled from the uniform (Haar) measure.

    Args:
        num_qubits: the number of qubits to generate.
        seed: An np.random.Generator (eg. np.random.default_rng())

    Returns:
        A random state sampled from the Haar measure
    """
    rng = RAND_SEED
    terms = 2 ** num_qubits
    # Random array over interval (0, 1]
    x = rng.random(terms)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.random(terms) * 2.0 * np.pi
    return np.sqrt(x / sumx) * np.exp(1j * phases)


def _evaluate_fidelity(unitary, thetas: List[float], k):
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
    return projection_norm_squared(state, k)


def _evaluate_objective_function(m_val, num_qubits, unitary, mu, sigm) -> float:
    """
    Implements equation 3
    """
    J = 0.0
    # Outer sigma
    for _ in range(m_val):
        # k is a quantum state which is a random sampled state
        k = _get_uniform_k(num_qubits)
        p_k = 1 / m_val

        avg = 0
        for _ in range(NUM_ROLLOUT):
            theta = _sample_gaussian_policy(mu, sigm)
            fid = _evaluate_fidelity(unitary, theta, k)
            avg += fid

        avg = avg / NUM_ROLLOUT

        J += p_k * avg

    return J


def _estimate_gradient(n_val, m_val, num_qubits, unitary, mu, sigma):
    """
    Implements equation 4 found on page 3
    """
    J_delta = np.zeros(n_val)

    for _ in range(m_val):
        # k is a quantum state which is a random sampled state
        k = _get_uniform_k(num_qubits)
        p_k = 1 / m_val

        avg = 0
        for _ in range(NUM_ROLLOUT):
            theta = _sample_gaussian_policy(mu, sigma)
            # Output of the ansatz circuit is the second term is Equation (3)
            fid = _evaluate_fidelity(unitary, theta, k)
            log_mu_gradient_estimate = _log_likelyhood_gradient_mu(mu, sigma, theta)
            avg += fid * log_mu_gradient_estimate

        avg = avg / NUM_ROLLOUT

        J_delta += p_k * avg

    return J_delta


def _log_likelyhood_gradient_mu(mu, sigma, thetas):
    return np.linalg.inv(sigma).dot((thetas - mu))


def _gradient_variance(previous_variance, current_gradient):
    return (GAMMA * previous_variance) + (
        (1 - GAMMA) * (np.square(current_gradient))
    )


def _step_and_optimize_mu(old_mu, previous_variance, mu_gradient):
    """
    This would be doing one step of optimizing mu after gradient estimation
    Implements equations (12) and (13)
    """
    new_variance = _gradient_variance(previous_variance, mu_gradient)
    new_mu = old_mu + (ETA * (mu_gradient / np.sqrt((new_variance + EPSILON))))
    return (new_mu, new_variance)


def pgrl_algorithm(num_qubits, unitary):
    """
    This is the high level algorithm which does policy gradient approach
    """

    # Define N as larger for more precision
    N_VAL = (2 * num_qubits - 1) * NUM_LAYERS

    # From page 10
    M_VAL = max(15 * num_qubits, num_qubits ** 2)

    mu = np.zeros(N_VAL)
    sigma = _get_covariance(N_VAL, 0)
    gradient_variance = np.zeros(N_VAL)

    J = []
    mus = []
    sigmas = []
    gradient_variances = []
    gradient_estimation = []

    for i in range(1, NUM_ITERATIONS):
        print(f"DEBUG: Iteration: {i}")

        J_step = []
        for _ in range(GRAPH_NUM):
            J_step.append(_evaluate_objective_function(M_VAL, num_qubits, unitary, mu, sigma))

        J.append(J_step)

        grad_est = _estimate_gradient(N_VAL, M_VAL, num_qubits, unitary, mu, sigma)
        gradient_estimation.append(grad_est)
        mu, gradient_variance = _step_and_optimize_mu(mu, gradient_variance, grad_est)
        mus.append(mu)
        sigma = _get_covariance(N_VAL, i)
        gradient_variances.append(gradient_variance)

    # At the very end have optimized mu, sigma
    return mus, sigmas, J, gradient_variances, gradient_estimation
