#######################################################################
#                              CONSTANTS                              #
#######################################################################

NUM_LAYERS = 5

MAX_ITER_REINFORCE = 2000

# From page 6 fig 6. description # TODO: revisit this?
T_VAL = MAX_ITER_REINFORCE

# Number of times to evaluate J at a certain step for graphing purposes
GRAPH_NUM = 10

EPSILON = 10e-8
GAMMA = 0.9
ETA = -0.5e-3