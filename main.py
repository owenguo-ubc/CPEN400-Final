from scipy.stats import unitary_group
from policy_gradient_rl import pgrl_algorithm
import matplotlib.pyplot as plt


def main():

    for i in range(5):
        # Choose a random unitary to approximate
        unitary = unitary_group.rvs(4)
        mu, sigma, J = pgrl_algorithm(unitary)

        # TODO: Currently just plotting the averages for each iteration
        #       try to setup the graph like the paper
        J = [ ( sum(x) / len(x) ) for x in J ]
        plt.plot(J)
        plt.savefig(f'run_{i}.png')
        plt.clf()


if __name__ == "__main__":
    main()