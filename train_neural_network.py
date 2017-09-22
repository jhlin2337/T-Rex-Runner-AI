import numpy as np
import constants
import pickle
import neural_network
from random import shuffle

# Trains the neural network using data stored inside constants.TRAINING_DATA_FILE and
# outputs the trained neural network parameters into the constants.NN_PARAMETERS_FILE.
def train_neural_network():
	training_data = np.load(constants.TRAINING_DATA_FILE)

	# num_data is the total number of data we have and num_train_data is the amount
	# we're going to use to train the neural network
	num_data = training_data.shape[0]
	num_train_data = int(num_data * constants.PERCENT_FOR_TRAINING)

	shuffle(training_data)

	# Create placeholders for inputs and labels
	X = np.zeros(shape=(num_data, constants.NN_INPUT_SIZE))
	Y = np.zeros(shape=(num_data, 1))

	# Retrieve data as X (neural net input) and Y (labels) from training_data
	for i in range(num_data):
		for j in range(constants.NN_INPUT_SIZE):
			X[i, j] = training_data[i][0][j]

	for i in range(num_data):
		Y[i, 0] = training_data[i][1]

	# Separate data into a training set and a test set
	X_train = X[0:num_train_data].T
	Y_train = Y[0:num_train_data].T
	X_test = X[num_train_data:num_data].T
	Y_test = Y[num_train_data:num_data].T

	# Train the neural network using the X_train and Y_train data
	parameters = neural_network.model(X_train, Y_train, X_test, Y_test)

	# Save the parameters for the neural network onto a file
	with open(constants.NN_PARAMETERS_FILE, 'wb') as f:
		pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)

train_neural_network()