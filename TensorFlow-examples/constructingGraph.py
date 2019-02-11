import tensorflow as tf
# There are two graphs, a default graph that is called when tensorflow is imported and 
# and the other that is created



## Script that shows what graph a tensor is on 

# g = tf.Graph()
# a = tf.constant(5)

# print(a.graph is g)
# print(a.graph is tf.get_default_graph())

## Script that shows default graphs when working with multiple graphs
# g1 = tf.get_default_graph()
# g2 = tf.Graph()

# print(g1 is tf.get_default_graph())

# with g2.as_default():
# 	print(g1 is tf.get_default_graph())

# print(g1 is tf.get_default_graph())


#First three nodes are told to output a constant value (5, 2, 3)
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

#The next two nodes get two existing variables as inputs and performs simple arithmetic operations on them

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)


#Run multiple nodes
with tf.Session() as sess:
	fetches = [a,b,c,d,e,f]
	outs = sess.run(fetches)

#A list containing the outputs of the nodes according to how they were ordered in the input list
print("outs = {}".format(outs))
print(type(outs[0]))
