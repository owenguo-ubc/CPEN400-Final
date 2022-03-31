#######################################################################
#                              CONSTANTS                              #
#######################################################################

# From page 6 fig 6. description
T_VAL = 500

# TODO set these
NUM_QUBITS = 4
NUM_LAYERS = 10

# Define n as larger for more precision
N_VAL = 2 * NUM_QUBITS * NUM_LAYERS

# From page 10
M_VAL = max(15 * NUM_QUBITS, NUM_QUBITS**2)

MAX_ITER_REINFORCE = 3

# Number of times to evaluate J at a certain step for graphing purposes
GRAPH_NUM = 10

EPSILON = 10e-8
GAMMA = 0.9
ETA = 3e-3