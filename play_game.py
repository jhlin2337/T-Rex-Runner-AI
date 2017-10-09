import tensorflow as tf
import numpy as np
import Quartz.CoreGraphics as CG
import pyautogui
import time
import constants
import neural_network
import pickle
from screencapture import capture_screen
from preparedata import rgbaImg2nnInput

# Jumps if the output for the forward propagation algorithm is positive and stays put otherwise.
# Jumping when nn_output is positive and staying put when nn_output is negative is equivalent to computing
# the activation, A3, of nn_output using a sigmoid function and jumping when A3 is > .5 and staying
# put when A3 is < .5
def perform_action(nn_output):
	time.sleep(constants.REFLEX_DELAY)
	if nn_output >= 0:
		pyautogui.press('up')
	return

# Loads the trained parameters, obtains pixel values for game, and plays the game
def play_game():
	# Load the parameters that was saved when we trained the neural network
	with open(constants.NN_PARAMETERS_FILE, 'rb') as f:
		parameters = pickle.load(f)

	# Creates the forward propagation tensorflow graph
	X = tf.placeholder(tf.float32, (constants.NN_INPUT_SIZE, 1))
	prediction = neural_network.forward_propagation(X, parameters)

	# Play the game
	with tf.Session() as sess:
		while True:
			# Capture pixel values on gamescreen in designated region and convert to
			# values that we can pass into our neural network
			gamescreen_region = CG.CGRectMake(
						constants.GAMESCREEN_X_ORIGIN, 
						constants.GAMESCREEN_Y_ORIGIN, 
						constants.GAMESCREEN_WIDTH, 
						constants.GAMESCREEN_HEIGHT)
			rgba_img = capture_screen(gamescreen_region)
			nnInput = rgbaImg2nnInput(rgba_img)

			# Predict and perform the next move
			nn_output = sess.run(prediction, feed_dict={X: nnInput})
			perform_action(nn_output[0][0])

play_game()

