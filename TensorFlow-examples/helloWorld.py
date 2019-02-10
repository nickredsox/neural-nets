#Hello World Example to check if Tensor Flow is installed

#Import tensorflow
import tensorflow as tf

print("Tensor Flow Version %s " % tf.__version__)

h = tf.constant("Hello")
w = tf.constant("World")

hw = h + w

print(hw)

with tf.Session() as sess:
	ans = sess.run(hw)

print(ans)