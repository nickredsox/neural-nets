#Softmax classification for MNIST dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#Constants
#Location to save the MNIST dataset
#
DATA_DIR = '/tmp/data'
NUM_STEPS = 1000
MINBATCH_SIZE = 100


#Read_data_sets downloads the dataset and saves it locally, and tells the utility how we want the data to be labelled
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

#x is image input. 784 pixels, 28x28 unrolled. None means we are not currently specifying how many of these images we will use at once
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))

#Elements representgin the true and predicted laels
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x,W)


#Measure of similarity chosen for the model (loss function)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels = y_true))

#How we are going to train the model. 0.5 is the learning rate controlling how fast 
#gradient descent optimizer shifts
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Once model is defined define the evaluation procdure. We are interested in the 
#fraction of the test that are correctly classified
correct_mask = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

	#Initialize all variables
	sess.run(tf.global_variables_initializer())

	#The actual training of the model in the gradient descent approach consists of taking 
	# steps in the right direction
	# feed dict
	for _ in range(NUM_STEPS):
		batch_xs, batch_ys = data.train.next_batch(MINBATCH_SIZE)
		sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})st

	#Test how well the model using test data that has never been seen by the model
	ans = sess.run(accuracy, feed_dict={x:data.test.images, y_true: data.test.labels})

print "Accuracy: {:.4}%".format(ans*100)