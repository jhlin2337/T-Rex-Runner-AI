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

# Number of nodes in each hidden layer of the neural network
HIDDEN_LAYER_1_SIZE = 25
HIDDEN_LAYER_2_SIZE = 12