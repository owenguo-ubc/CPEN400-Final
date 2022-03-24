import pennylane as qml
import numpy as np
from typing import List


def _policy_gradient_descent(vqa_qnode) -> List[float]:
    """
    TODO:
    """
    return None


def _build_V(num_qbits, thetas):
    """
    TODO:
    """
    pass


def build_vqa_qnode(unitary) -> qml.QNode:
    num_qbits = np.log2(unitary.shape[0])

    dev = qml.device("default.qubit", wires=num_qbits, shots=1)

    @qml.qnode(dev)
    def qnode(k, thetas):
        # Assuming k is sampled from the computation basis
        for index, b in enumerate(k):
            if b == "1":
                qml.PauliX(wires=index)

        qml.QubitUnitary(unitary, wires=range(num_qbits))
        _build_V(num_qbits, thetas)

        # TODO: return current state |ψ⟩ projected onto |k〉
        return

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
    return proj * np.conj(proj)
    
def pg_vqa_compile(unitary) -> List[float]:
    vqa_qnode = build_vqa_qnode(unitary)

    thetas = _policy_gradient_descent(vqa_qnode)

    return thetas
