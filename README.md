# CPEN 400 Final Project

Link to paper [Policy Gradient Approach to Compilation of Variational Quantum Circuits](https://arxiv.org/pdf/2111.10227.pdf)

## Usage

A test script, `five_qubit_test.py`, is provided to run the PGRL algorithm on the five qubit case. Ensure the required Python modules are installed as defined in `requirements.txt` and run the following:

```
python3 five_qubit_test.py
```

> Note that the default number of iterations for the algorithm is set at 1000, which may take several hours. This value can be changed in `src/constants.py`.

### Ansatz Circuit Generation Demo

The printout of the the ansatz circuit creation can be see by running the `policy_gradient_vqa.py` script. The script provides a CLI interface:


```
$ python3 src/policy_gradient_vqa.py --help

usage: policy_gradient_vqa.py [-h] [--num_qubits NUM_QUBITS] [--num_layers NUM_LAYERS]

PGRL Ansatz Debug

optional arguments:
  -h, --help            show this help message and exit
  --num_qubits NUM_QUBITS
                        How many qubits should we create the test unitary for
  --num_layers NUM_LAYERS
                        How many layers should be in the anzatz
```

### Team Contributions

Owen: Wrote reinforcement learning algorithm

Akhil: Wrote dynamic ansatz circuit creation

Ross: Wrote code for evaluating multiple parts needed by reinforcement algorithm

Kobe: Wrote code for sampling random states and projection

All: Companion report and presentation slides

Note that the github commit history might not be accurate since a lot of the work was done on visual studio code live share via pair programming
