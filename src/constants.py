#######################################################################
#                              CONSTANTS                              #
#######################################################################

NUM_LAYERS = 10

MAX_ITER_REINFORCE = 3000

# From page 6 fig 6. description # TODO: revisit this?
T_VAL = MAX_ITER_REINFORCE

# Number of times to evaluate J at a certain step for graphing purposes
GRAPH_NUM = 10

EPSILON = 10e-8
GAMMA = 0.9
ETA = 3e-3
