
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import image



def deconstruct_image(image):


def reconstruct_image():
	return 0

tf.set_random_seed(1)
np.random.seed(1)


zzz = image.run()
# print(color_intensities)

# zzz = [ 3.45591175,  4.21306292,  1.39260529,  0.79979209,  2.80560091,  1.07702818,
#   1.13857619,  2.06239613,  1.01650569,  2.74965652,  2.89521513,  1.78206757,
#   3.03480862,  1.155502,    0.84656923,  0.94706706,  1.30763502, 14.22199015,
#   1.01494266,  0.82881535,  1.20278014,  2.6748784,   1.52933522,  0.93917355]



# gradient descent
# -1/1 8 0.1 : 0.002077
# -1/1 8 0.5 : 0.001934 then waver up and down
# -1/1 8 0.8 : 
# -1/1 8 1.0 : 12.37966 .....

# adamoptimizer
# -1/1 10 0.1 : 0.022
# -1/1 10 0.001 : 0. 001126
# -1/1 10 0.0005 : 0. 0011261

# -1/1 16 0.0005 : 0.001918

# -1/1 20 0.001 : 0.001126
# -1/1 20 0.0005 : 0.000765
# -1/1 20 0.0001 : 0.0007651
# -1/1 25 0.00005 : 0.0007651

# 0/1 20 0.0001 : 0.0168
# -2/1 20 0.0001 : 0.0008762


# -0.5/0.5 20 0.0001 : 0.00109

# -2.0/2.0 20 0.0001 : 0.0016359

# -1/1 25 0.0005 : 0.0016355
# -1/1 30 0.0005 : 0.0016355

color_intensities = zzz / np.linalg.norm(zzz)

# image -> color intensity 

# provided color_intensity -> avg of your color intensities

# e



# fake data
x = np.linspace(-1.0, 1.0, 25)[:, np.newaxis]        # shape (100, 1)
# noise = np.random.normal(0, 0.1, size=x.shape)
y = np.asarray(color_intensities)
y.shape=(25,1)                       # shape (100, 1) + some noise

# plot data
# plt.scatter(x, y)
# plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

# step = tf.Variable(0, trainable=False)
# rate = tf.train.exponential_decay(0.0005, step, 1, 0.9999)

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.AdamOptimizer(0.0005)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ioff()   # something about plotting

for step in range(25000):
	# train and net output
	_, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
	if step % 1000 == 0:
		# plot and show learning process
		# plt.cla()
		# plt.scatter(x, y)
		# plt.plot(x, pred, 'r-', lw=5)
		# plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
		# plt.pause(1.0)
		# plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
		# plt.show()
		print('Loss=%.12f' % l)

tvars = tf.trainable_variables()
tvars_vals = sess.run(tvars)

for var, val in zip(tvars, tvars_vals):
    print(var.name, val)  # Prints the name of the variable alongside its value.

# print(output.numpy())

plt.scatter(x, y)
plt.plot(x, pred, 'r-', lw=2)
plt.text(0.5, 0, 'Loss=%.6f' % l, fontdict={'size': 20, 'color': 'red'})
plt.pause(1.0)
plt.text(0.5, 0, 'Loss=%.6f' % l, fontdict={'size': 20, 'color': 'red'})
plt.show()

# plt.ioff()
plt.show()