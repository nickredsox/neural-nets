import tensorflow as tf


a = tf.constant(5)
b = tf.constant(2)

c = tf.multiply(a,b)
d = tf.add(a,b)
e = tf.subtract(d,c)
f = tf.add(c,d)
g = tf.divide(f,e)

#c = a*b = 5*2 = 10
#d = a+b = 5+2 = 7
#e = d-c = 7-10 = -3
#f = c+d = 10 + 7 = 17
#g = f/e = 17/-3 = -5.6667

#launch the graph in a tf.Session
#a session is part of Tensorflow API that communicates between Python objects and data on our end
sess = tf.Session()

#Execution - completes one set of cimputations in graph
# starts at the requested outuputs and works backwards
#
# We requested that node f be computed
outs = sess.run(g)

#Close the sessions to free up resources
sess.close()

print("outs = {}".format(outs))