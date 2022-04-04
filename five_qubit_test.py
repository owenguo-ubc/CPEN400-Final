from scipy.stats import unitary_group
import pennylane as qml
import sys
from src.policy_gradient_rl import pgrl_algorithm
import matplotlib.pyplot as plt
from src.constants import NUM_LAYERS
from src.policy_gradient_vqa import projection_norm_squared, _create_anzatz_circuit
from src.policy_gradient_rl import _get_uniform_k
import numpy as np

# Test constants
NUM_QUBITS = 2

# Choose a random unitary to approximate
unitary = unitary_group.rvs(2 ** NUM_QUBITS)

# Sanity check projection
def build_vqa_qnode(unitary) -> qml.QNode:
    num_qbits = int(np.log2(unitary.shape[0]))

    dev = qml.device("default.qubit", wires=num_qbits, shots=1)

    @qml.qnode(dev)
    def qnode(k, n_layers):
        # Assuming k is sampled from the computation basis
        qml.QubitStateVector(k, wires=range(num_qbits))

        qml.QubitUnitary(unitary, wires=range(num_qbits))
        qml.QubitUnitary(np.matrix(unitary).getH(), wires=range(num_qbits))

        return qml.state()

    return qnode


# Call QNode from policy_gradient_vqa.py
k = _get_uniform_k(NUM_QUBITS)
qnode = build_vqa_qnode(unitary)
state = qnode(k, NUM_LAYERS)

# Project the resulting state from calling qnode onto |k>
assert projection_norm_squared(state, k) > 0.99

# Sanity check anzatz
_create_anzatz_circuit(5, range(10 * 10), 10)

# Run the policy gradient algorithm
mus, sigma, J = pgrl_algorithm(NUM_QUBITS, unitary)

# Plot the results
plot_avg = []
plot_high = []
plot_low = []
plot_x = []

for idx, iteration in enumerate(J):
    plot_avg.append(sum(iteration) / len(iteration))
    plot_high.append(max(iteration))
    plot_low.append(min(iteration))
    plot_x.append(idx)

plt.plot(plot_x, plot_avg)
plt.fill_between(plot_x, plot_high, plot_low, color="#97b5d4")
plt.savefig(f"five_qubit_result.png")
plt.clf()

plt.plot(plot_x, mus)
plt.savefig(f"mus.png")
plt.clf()
