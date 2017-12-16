# Build Deep Conv Network for MNIST: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Get the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Will set up a convolutional network with layers:
# conv --> pool --> conv --> pool --> fc --> dropout --> fc

# In general initalize weights/bias with small amount of noise, w/ relu, give slight positive bias
# Create functions to make weights and variables
# tf.truncated normal outputs values from a truncated normal distribution
# Takes arg shape which is the shape of the output, stddev which is standard deviation of the normal
# Return a .Variable
# Weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# CONVOLVE FUNCTION
# Will use stride = 1, padding = 'SAME' which will ensure the output of conv layer is same shape, outputHeight = (heightIn - filterHeight)/stride + 1
# tf.nn.conv2d takes 1st arg input, will be the images, 4 dimensions 
# 2nd arg filter, same type as input  w/ 4 dimensions: height, width, in channels, out channels
# 3rd arg strides, a list of ints, 1D tensor with length 4, stride of sliding window for each dimension of input
# 4th arg padding, zero padding to border the image, either 'SAME' or 'VALID'. Use same to keep same shape
# there are other args but not using em
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# MAX POOLING FUNCTION
# Will use max pooling, 2x2 window, strid of 2
# tf.nn.max_pool takes 1st arg value, 4D tensor, will be the convolutional output
# 2nd arg ksize, 1D tensor of 4 elements, size of window for each dimension of input
# 3rd arg stride, 1D tensor of 4 elements, stride of window for each dimension of input (stride, last one was size)
# 4th arg padding, same as before
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
  
  
# Create symbolic variable/placeholder for input. Input is 28x28 image or 784 flattened array
x = tf.placeholder(tf.float32, shape=[None, 784])

# Create placeholder for labels, correct answers
y = tf.placeholder(tf.float32, shape=[None, 10])


# FIRST CONVOLUTIONAL LAYER
# image --> conv --> pool
# Use our functions to make weight variables
  
# Set Weights
# weights are 5x5 window, input channel is 1 (not 3 color channels), output channels, num of filters
W_conv1 = weight_variable([5, 5, 1, 32]) # height, width, input Channel Num, outputCHannel Num/numFilters

# Set Bias
# bias is straight vector, 32 filters so make bias 32 long
b_conv1 = bias_variable([32]) # only arg is length of bias vector

# Set input
# Need to reshape the image to the right shape
# 28x28 is height, width, 1 at the end is number of color channels
x_image = tf.reshape(x, [-1, 28, 28, 1]) 

# Do the actual convolution, put it through relu activation
# Use our conv2d function we made which takes input, weight
# tf.nn.relu does the rectified relu, linear if > 0, 0 if < 0
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Do the actual pooling
# Give the output of convolution to our pooling function
# Reduces the image size to 14 14 since outputHeight = (inputHeight - filterHeight)/stride + 1 = (28-2)/2 + 1 = 14
# The depth will be the same, 32 since there were 32 filters
h_pool1 = max_pool_2x2(h_conv1)



# SECOND CONVOLUTIONAL LAYER
# Set Weights
# Filter size is still 5x5, input channel num is 32 since thats the depth coming out of the first layer
# last layer it was 1 since the image only had 1 color (grayscale) channel
# This layer we have 64 filters  so outputChannel arg is 64
W_conv2 = weight_variable([5, 5, 32, 64])


# Set Biases
# Same thing, biases are a flat vector, same length as depth of weights, 64
b_conv2 = bias_variable([64])


# Actual Convolution
# Same deal, use our function, put it into relu activation
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Actual Pooling 
# Use pooling function with conv output
# Conv output kept the shape so it was 14x14
# After this pooling we will have (14-2)/2 + 1 = 7x7
# Depth stays 64
h_pool2 = max_pool_2x2(h_conv2)



# FULLY CONNECTED LAYER
# Add a fully connected layer, like how most neural network layers are
# Set Weights
# Give it 1024 neurons so we will output 1024 x n
# Input to the layer is volume 7 x 7 x 64
# Shape of the weight will be volume x 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])

# Set Biases
# Length of biases is the same as 1024 depth
b_fc1 = bias_variable([1024])

# Reshape output from the pooling layer / input to this layer
# Flatten volume
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# Forward Pass
# matrix multiply flattened pool x Weight so [-1 x 3136] x [3136 x 1024], add the 1024 bias
# so we get 1024 as the output dimension of this layer
# Also put it through relu after
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



# DROPOUT LAYER
# Make sure network doesnt find just 1 path through/ reduce overfitting
# Create a tf placeholder for probability that a neuron's output is kept
# With placeholder we can turn dropout on for training, off for testing
keep_prob = tf.placeholder(tf.float32)

# The actual dropout, tf.nn.dropout args are output of fully connected, probabiltiy of keeping neuron output
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# CLASSIFICATION LAYER
# Softmax for final layer
# Input is [n x 1024]
# Wanted output is [n x 10]
# Weights will be [1024 x 10]
# Set Weights
W_fc2 = weight_variable([1024, 10])

# Set Biases
# Same length as num columns, 10
b_fc2 = bias_variable([10])


# Forward Pass through layer
# Do input * weight  + bias
# So matrix multiply [n x 1024] x [1024 x 10] to get an nx10 output shape
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



# the softmax_cross_entropy with logits will internally apply the softmax
# on the y_conv and sums across the classes.
# reduce_mean takes average over the sums
cross_entropy = tf.reduce_mean(  tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)  )

# Replace the gradient descent with an adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Same as before, list of True or Falses
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

# Same as before, average of the correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#with tf.Session() as sess:
sess = tf.InteractiveSession()
  
# Initialize variables
sess.run(tf.global_variables_initializer())
  
  # Num Passes
for i in range(100):
   # print(i)

    # get batch of images, labels
    batch = mnist.train.next_batch(50)

    # every once in a while look at training accuracy
    if i % 100 == 0:
    
        # Get accuracy on current batch, keep prob is 1 so we dont use drop out
        train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

    # Run optimizer on current batch, dropout at 50%
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

  # After training get test accuracy, dont use dropout
  #print('test accuracy %g' % accuracy.eval(feed_dict={
   #   x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    
    
# PRINT IMAGE WITH PREDICTION
index = np.random.randint(10000)
im = mnist.test.images[index,:]
im = im.reshape(1,784)

predictions = sess.run(y_conv, feed_dict ={ x:im, keep_prob:1.0})
prediction = int(sess.run(tf.argmax(predictions,1)))

im = im.reshape(28,28)
plt.imshow(im, cmap = 'binary')
title = 'Prediction: ' + str(prediction)
plt.title(title)










  

