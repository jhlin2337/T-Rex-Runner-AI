import numpy as np
import tensorflow as tf
import constants
from tensorflow.python.framework import ops

# Uses a xavier initializer to create the starting weights for the neural network and initialize 
# all biases to zero. Return the weights and biases in a dictionary
def initialize_parameters():
    # Initialize weights and biases
    l1_weights = tf.get_variable("l1_weights", [constants.HIDDEN_LAYER_1_SIZE, constants.NN_INPUT_SIZE], initializer = tf.contrib.layers.xavier_initializer())
    l1_biases = tf.get_variable("l1_biases", [constants.HIDDEN_LAYER_1_SIZE, 1], initializer = tf.zeros_initializer())
    l2_weights = tf.get_variable("l2_weights", [constants.OUTPUT_LAYER_SIZE, constants.HIDDEN_LAYER_1_SIZE], initializer = tf.contrib.layers.xavier_initializer())
    l2_biases = tf.get_variable("l2_biases", [constants.OUTPUT_LAYER_SIZE, 1], initializer = tf.zeros_initializer())

    # Save weights and biases onto a dictionary
    parameters = {"l1_weights": l1_weights, "l1_biases": l1_biases, "l2_weights": l2_weights, "l2_biases": l2_biases}
    
    return parameters

# Given a dataset containing the input for the neural network <X> and the parameters for the
# neural network <parameters>, returns the value of the output node for the neural network after
# forward propagation. Note that the sigmoid function has not yet been applied to the output node
def forward_propagation(X, parameters):
    hidden_layer = tf.add(tf.matmul(parameters['l1_weights'], X), parameters['l1_biases'])
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.add(tf.matmul(parameters['l2_weights'], hidden_layer), parameters['l2_biases'])
    
    return output_layer


# Given two sets of data, one for training and the other for testing, implements a 
# shallow, two-layer neural network. This function returns a dictionary containing the 
# weights and biases that the model learned from the training set.
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 2000, batch_size = 50):

    ops.reset_default_graph()

    # Initialize relevant variables
    num_train_examples = X_train.shape[1]
    input_size = X_train.shape[0]
    output_size = Y_train.shape[0]
    
    X = tf.placeholder(tf.float32, shape=(input_size, None))
    Y = tf.placeholder(tf.float32, shape=(output_size, None))

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Tensorflow graph for forward propagation
    prediction = forward_propagation(X, parameters)
    
    # Tensorflow graph for the cost function
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(prediction), labels=tf.transpose(Y)))
    
    # Back propagation using Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize variables
    init = tf.global_variables_initializer()

    # Start session
    with tf.Session() as sess:
        
        # Run initialization
        sess.run(init)
        
        # Train the network
        for epoch in range(num_epochs):
            
            epoch_cost = 0.
            batch_index = 0

            while batch_index < num_train_examples:
                start = batch_index
                end = batch_index+batch_size
                X_batch = np.array(X_train[0:None, start:end])
                Y_batch = np.array(Y_train[0:None, start:end])

                _, c = sess.run([optimizer, cost], feed_dict={X: X_batch, Y: Y_batch})

                epoch_cost += c
                batch_index += batch_size

            if epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

        # Save parameters
        parameters = sess.run(parameters)

        # Calculate accuracy
        correct = tf.equal(tf.round(tf.sigmoid(prediction)), Y)
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))

        # Print accuracy on training set and testing set
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters