import pennylane as qml
from scipy.stats import unitary_group
import numpy as np
import argparse
from typing import List


def _create_anzatz_circuit(n_wires, thetas, n_layers=1):
    """Create layers of single qubit aqnd two qubit rotations that
    create maximal overlap.

    There are two main cases to handle:

    1. An odd number of thetas where the second and third column of two qubit
    gates match. The third columb is just offset 1 down.

          |          Layer 1         |     .........    |      Layer N

          ┌──┐   ┌──┐
          │θ1├───┤  ├─────────────
          └──┘   │  │
                 │θ4│
          ┌──┐   │  │   ┌──┐
          │θ2├───┤  ├───┤  ├──────
          └──┘   └──┘   │  │
                        │θ5│
          ┌──┐          │  │
          │θ3├──────────┤  ├──────
          └──┘          └──┘

    2. An even number of thetas where the second column has 1 more 2 qubit gate.


          |          Layer 1         |     .........    |      Layer N

          ┌──┐   ┌──┐
          │θ1├───┤  ├────────────────
          └──┘   │  │
                 │θ5│
          ┌──┐   │  │   ┌──┐
          │θ2├───┤  ├───┤  │
          └──┘   └──┘   │  │
                        │θ7│  ...
          ┌──┐   ┌──┐   │  │
          │θ3├───┤  ├───┤  │
          └──┘   │  │   └──┘
                 │θ6│
          ┌──┐   │  │
          │θ4├───┤  ├───────────────
          └──┘   └──┘

    The thetas are applied in column order.

    :param n_wires: The number of wires to create this anzatz circuit
    :param thetas: Thetas to construct the anzatz circuit
    :param n_layers: How many layers we need.

    """
    θ_counter = 0

    for _ in range(n_layers):

        # Create single qubit column
        for w in range(n_wires):
            qml.RY(thetas[θ_counter], wires=[w])
            θ_counter += 1

        # Look at ever other wire
        for w in range(n_wires)[:-1:2]:
            _apply_rzz(thetas[θ_counter], [w, w + 1])
            θ_counter += 1

        # Look at ever other wire, offset by 1, skipping the last one
        for w in range(n_wires)[1:-1:2]:
            _apply_rzz(thetas[θ_counter], [w, w + 1])
            θ_counter += 1

    return θ_counter

def _apply_rzz(theta, wires):
    """Double Qubit gates

    :param theta: The 2-Qubit theta
    :param wires: The wires

    """
    qml.MultiRZ(theta, wires=wires)


def build_vqa_qnode(unitary) -> qml.QNode:
    num_qbits = int(np.log2(unitary.shape[0]))

    dev = qml.device("default.qubit", wires=num_qbits, shots=1)

    @qml.qnode(dev)
    def qnode(k, thetas, n_layers):
        # Assuming k is sampled from the computation basis
        qml.MottonenStatePreparation(state_vector=k, wires=range(num_qbits))

        qml.QubitUnitary(unitary, wires=range(num_qbits))
        _create_anzatz_circuit(num_qbits, thetas, n_layers)

        return qml.state()

    return qnode


def projection_norm_squared(a, b):
    """
    Takes two state vectors a, and b (in the format returned from qml.state)
    and calculates |<a|b>|^2
    """
    # Convert |a> to <a|
    a = a.conjugate()
    # Calculate <a|b>
    proj = np.dot(a, b)
    # return |<a|b>|^2
    return (proj * np.conj(proj)).real


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quiz3")
    parser.add_argument(
        "--num_qubits",
        help="How many qubits should we create the test unitary for",
    )
    parser.add_argument(
        "--num_layers",
        help="How many layers should be in the anzatz",
    )
    args = parser.parse_args()
    size = 2 ** (int(args.num_qubits))
    layers = int(args.num_layers)
    U = unitary_group.rvs(2 ** int(args.num_qubits))

    RAND_SEED = np.random.default_rng()

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

    print(
        qml.draw(build_vqa_qnode(U))(
             _get_uniform_k(int(args.num_qubits)), range(layers * 2 * (int(args.num_qubits))), n_layers=layers
        )
    )
