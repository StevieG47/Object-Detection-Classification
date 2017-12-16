# Tensorflow MNIST tutorialL https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf

# Get the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Create symbolic variable/placeholder for input. Input is 28x28 image or 784 flattened array
# None means this dimension can be any length. It will be number of training images
x = tf.placeholder(tf.float32, [None, 784]) # type, shape


# Create Variables for weights and biases. .Variable is a modifiable tensor, usually make model parameters Variables
# We do Wx + b so [n x 784][784 x 10] + [10 x 1] = [n x 10] output shape
# Initialize as zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# Forward Pass through layer
# tf.matmul(x,W) does matrix multiplication,
# wx + b goes into the softmax with tf.nn.softmax()
yhat = tf.nn.softmax(tf.matmul(x, W) + b)


# Create placeholder for labels, correct answers
y = tf.placeholder(tf.float32, [None, 10]) # None used again to represent n number of training images


# Loss is cross entropy, L = - sum y*log(yhat)
# Use tf.log for logarithm
# tf.reduce_sum is sum of elements in the reduced_indices dimension, so like the first and first, the second and second
# if the label were [1 2 3] and the prediction was [4 5 6] then tf.reduce_sum = 5+7+9 = 21
# reduction_indices argument tells it which axis to sum
# tf.reduce_mean gets the mean over all examples in the batch, so we get a mean sum 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))


# Backprop and update weights done for us w/ tensorflow


# Need to pick optimizer to modify variables, reduce loss
# .GradientDescentOptimizer uses gradient descent to minimize the argument cross_entropy, our loss
# argument of GradientDescentOptimizer is learning rate, set to .5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Ok cool we created the graph now launch the model in an interactive session
sess = tf.InteractiveSession()


# Operation to initialize variables
tf.global_variables_initializer().run()


# The actual training
# Use batch size of 100
# 1000 passes
# use sess.run to use our train step, setting our x and y variables to the batch variables
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})


# EVALUATE MODEL

# Using one hot encoding so take max probability from 10 classes
# tf.argmax is just an argmax, returns indices of max. the 1 tells it the axis,
# since each row is a different image, do argmax horizontally across all columns (10 columns for each class, take highest probability)
# Get predicted class do tf.argmax(yhat,1), Get actual class do tf.argmax(y,1)
# tf.equal gives a list of booleans, depending on if it was equal or not
correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(y,1))


# Get accuracy by taking mean of correct_prediction which is just a list of [1 0 1 1 0 ] etc.
# Taking mean will give us $ accuracy
# tf.cast takes 1st argument and makes it datatype 2nd argument. Since list was [True False ...] 
# and we want numbers cast to floats
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Print Accuracy
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) # geed_dict sets those placeholders we defined
print('Accuracy: ', round(acc*100,2), '%')








