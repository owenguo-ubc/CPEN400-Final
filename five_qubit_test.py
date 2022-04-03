from scipy.stats import unitary_group
from src.policy_gradient_rl import pgrl_algorithm
import matplotlib.pyplot as plt

# Test constants
NUM_QUBITS = 5

# Choose a random unitary to approximate
unitary = unitary_group.rvs(2 ** NUM_QUBITS)

# Run the policy gradient algorithm
mu, sigma, J = pgrl_algorithm(NUM_QUBITS, unitary)

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
