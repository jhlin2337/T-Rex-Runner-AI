# Initialize the coordinates and dimensions of the gamescreen
GAMESCREEN_X_ORIGIN = 0
GAMESCREEN_Y_ORIGIN = 96
GAMESCREEN_WIDTH = 400
GAMESCREEN_HEIGHT = 200

# Dimensions of computer screen
SCREENWIDTH = 1440
SCREENHEIGHT = 900

# Names of relevant files being used
TRAINING_DATA_FILE = 'training_data.npy'
NN_PARAMETERS_FILE = 'nn_params.pkl'

# Size for thumbnail and size for input for neural network
THUMBNAIL_WIDTH = 80
THUMBNAIL_HEIGHT = 40
NN_INPUT_SIZE = THUMBNAIL_WIDTH * THUMBNAIL_HEIGHT

# Proportion of data to be allocated for training
PERCENT_FOR_TRAINING = .9

# Number of nodes in each layer of the neural network
HIDDEN_LAYER_1_SIZE = 1024
# HIDDEN_LAYER_1_SIZE = 32
# HIDDEN_LAYER_2_SIZE = 16
OUTPUT_LAYER_SIZE = 1

# A number in seconds that accounts for the reflex time
# of the person who created the training data. The program
# will wait for this long after its decided what move to make
REFLEX_DELAY = .1