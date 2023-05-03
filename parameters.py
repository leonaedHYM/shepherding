# Source:
# Strombom D, Mann RP,
# Wilson AM, Hailes S, Morton AJ, Sumpter DJT,
# King AJ. 2014 Solving the shepherding
# problem: heuristics for herding autonomous,
# interacting agents. J. R. Soc. Interface 11:
# 20140719.
# http://dx.doi.org/10.1098/rsif.2014.071

FOLDER_PATH = 'multi_model3_newNetwork_newEnv_oldReward'
MODEL_PATH = '55020multi_model_linear4.pth'
# output/render while training (set to False when training on beast GPU server)
BATCH_SIZE=64
OUTPUT = True       # output detailed game state information to terminal
RENDER = True    # render pygame
USER_INPUT = False  # seed training with user input from keyboard
FRAME_RESET = 500   # automatically reset game after x frames
SAVE_TARGET = 60    # save and update target after x games
CNN_NETWORK = False     # alternatively, use linear network
SUCCESS_METRICS = 0.1

FIELD_LENGTH = 100      # field width and height
MAX_NUM_AGENTS = 3    # number of agents [0, MAX_NUM_AGENTS]
NUM_SHEPHERDS = 2
R_S = 15       # shepherd detection distance
R_R = 3    # agent repulsion distance
R_A = 10   # agent attraction distance
R_P = 15   # sheep's perception distance
TARGET_RADIUS = 20     # distance from target to trigger win condition
# P_A > P_C > P_S (eliminate inertial term for discrete setting)
P_A = 2     # agent repulsion weight
P_C = 1.2    # LCM attraction weight
P_S = 1     # shepherd repulsion weight