import tensorflow as tf


#First three nodes are told to output a constant value (5, 2, 3)
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

#The next two nodes get two existing variables as inputs and performs simple arithmetic operations on them

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

# Node d multiplies the outputs of a and b
# Node e adds the output of b and c
# Node f subtracts the output of e from d

#d = a*b = 5*2 = 10
#e = c+b = 3+2 = 5
#f = d-e = 10-5 = 5

#launch the graph in a tf.Session
#a session is part of Tensorflow API that communicates between Python objects and data on our end
sess = tf.Session()

#Execution - completes one set of cimputations in graph
# starts at the requested outuputs and works backwards
#
# We requested that node f be computed
outs = sess.run(f)

#Close the sessions to free up resources
sess.close()

print("outs = {}".format(outs))