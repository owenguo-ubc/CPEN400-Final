#######################################################################
#                              CONSTANTS                              #
#######################################################################

# Number of layers in the Ansatz circuit
NUM_LAYERS = 4

# Number of iterations to run PGRL
NUM_ITERATIONS = 1000

# Max iteration constant
MAX_ITER_REINFORCE = 3000

# Number of times to evaluate J at a certain step for graphing purposes
GRAPH_NUM = 10

# RMSProp constants
T_VAL = MAX_ITER_REINFORCE
EPSILON = 10e-8
GAMMA = 0.9
ETA = -0.5e-3
