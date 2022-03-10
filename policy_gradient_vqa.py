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


def pg_vqa_compile(unitary) -> List[float]:
    vqa_qnode = build_vqa_qnode(unitary)

    thetas = _policy_gradient_descent(vqa_qnode)

    return thetas
