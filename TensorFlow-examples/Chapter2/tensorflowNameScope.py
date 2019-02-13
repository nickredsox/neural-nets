import tensorflow as tf

# every name in a graph must be unique. 
# This example creates two graphs and names every element 'c'
# Tensorflow will automatically rename a node with an underscore if the same # name occurs in the same graph
# so here the output will be c:0
# prefix_name/c:0
# prefix_name/c_1:0


with tf.Graph().as_default():
	c1 = tf.constant(4,dtype=tf.float64,name='c')
	with tf.name_scope("prefix_name"):
		c2 = tf.constant(4, dtype=tf.int32,name='c')
		c3 = tf.constant(4, dtype=tf.int32,name='c')

print(c1.name)
print(c2.name)
print(c3.name)