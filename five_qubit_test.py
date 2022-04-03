from scipy.stats import unitary_group
from src.policy_gradient_rl import pgrl_algorithm
import matplotlib.pyplot as plt

# Test constants
NUM_QUBITS = 4

unitary = unitary_group.rvs(2 ** NUM_QUBITS)
mu, sigma, J = pgrl_algorithm(NUM_QUBITS, unitary)

# TODO: Currently just plotting the averages for each iteration
#       try to setup the graph like the paper
J = [ ( sum(x) / len(x) ) for x in J ]
plt.plot(J)
plt.savefig(f'five_qubit_result.png')
plt.clf()