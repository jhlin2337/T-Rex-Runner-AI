import pyautogui
import numpy as np
import Quartz.CoreGraphics as CG
import os
import time
import constants
from curtsies import Input
from screencapture import capture_screen
from preparedata import rgbaImg2nnInput

# Returns:
# training_data - a list containing the training data inside the training
# 				  data file. If no such file exists, training_data will be
#				  set to an empty list.
def open_training_data_file():
	if os.path.isfile(constants.TRAINING_DATA_FILE):
		training_data = list(np.load(constants.TRAINING_DATA_FILE))
	else:
		training_data = []

	return training_data

# Arguments:
# seconds - integer representing the time duration of the countdown
def countdown(seconds = 3):
	for i in range(seconds):
		print(seconds-i)
		time.sleep(1)

# Returns:
# isPressed - 1 if up key was pressed, 0 otherwise
def up_arrow_pressed():
	with Input(keynames='curses') as input_generator:
		for e in input_generator:
			key = repr(e)[1:len(repr(e))-1]
			if key == 'KEY_UP':
				return 1
			break
	return 0

# Arguments:
# training_data - list of previous training data that the function appends
#				  new training data onto
def record_data(training_data):
	while(True):
		gamescreen_region = CG.CGRectMake(
			constants.GAMESCREEN_X_ORIGIN, 
			constants.GAMESCREEN_Y_ORIGIN, 
			constants.GAMESCREEN_WIDTH, 
			constants.GAMESCREEN_HEIGHT)
		rgba_img = capture_screen(gamescreen_region)
		nnInput = rgbaImg2nnInput(rgba_img)
		output = up_arrow_pressed()
		training_data.append([nnInput, output])
		print(len(training_data))

		if len(training_data) % 50 == 0:
			print('Saving Data...')
			np.save(constants.TRAINING_DATA_FILE, training_data)

def main():
	training_data = open_training_data_file()
	countdown()
	record_data(training_data)

main()


