from scipy.stats import unitary_group
from src.policy_gradient_rl import pgrl_algorithm
import matplotlib.pyplot as plt

# Test constants
NUM_QUBITS = 5

# Choose a random unitary to approximate
unitary = unitary_group.rvs(2**NUM_QUBITS)

# Run the policy gradient algorithm
mus, sigmas, J, gradient_variances, gradient_estimation = pgrl_algorithm(NUM_QUBITS, unitary)

# Sort the objective function results to plot the spread and the mean
plot_avg = []
plot_high = []
plot_low = []
plot_x = []

for idx, iteration in enumerate(J):
    plot_avg.append(sum(iteration) / len(iteration))
    plot_high.append(max(iteration))
    plot_low.append(min(iteration))
    plot_x.append(idx)

# Plot the objective function results like in the paper
plt.plot(plot_x, plot_avg)
plt.fill_between(plot_x, plot_high, plot_low, color="#97b5d4")
plt.savefig(f"five_qubit_result.png")
plt.clf()

# Auxillary debug plots
plt.plot(plot_x, mus)
plt.savefig(f"mus.png")
plt.clf()

plt.plot(plot_x, gradient_variances)
plt.savefig(f"gradient_variances.png")
plt.clf()

plt.plot(plot_x, gradient_estimation)
plt.savefig(f"gradient_estimation.png")
plt.clf()
