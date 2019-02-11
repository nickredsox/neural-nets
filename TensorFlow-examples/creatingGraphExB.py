import tensorflow as tf

#Figure 3-3 B

a = tf.constant(5)
b = tf.constant(2)


#to_float(num) --> converts output to a float
c = tf.multiply(a,b)
d = tf.sin(tf.to_float(c))
e = tf.divide(tf.to_float(b),d)

#c = a*b = 5*2 = 10
#d = sin(c)= sin(10) = -0.544
#e = b/d = 2/-0.544 = -3.67

#launch the graph in a tf.Session
#a session is part of Tensorflow API that communicates between Python objects and data on our end
sess = tf.Session()

#Execution - completes one set of cimputations in graph
# starts at the requested outuputs and works backwards
#
# We requested that node e be computed
outs = sess.run(e)

#Close the sessions to free up resources
sess.close()

print("outs = {}".format(outs))