
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,  reshape=False, validation_size=0)
np.set_printoptions(threshold=np.nan)

# for holding data of the image pixels
# None = No value provided for no. of rows
# 28,28 = size of pixel
# 1 = no. of values for each pixel
x = tf.placeholder(tf.float32,[None,28,28,1])

# Placeholder for all known probabilities of classification
y_ = tf.placeholder(tf.float32, [None,10])
pkeep = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

K = 6
L = 12
M = 24
N = 200


# Variable to store values decided by TF
w1 = tf.Variable(tf.truncated_normal([6,6,1,K],stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,tf.float32,[K]))

w2 = tf.Variable(tf.truncated_normal([5,5,K,L],stddev=0.1))
b2 = tf.Variable(tf.constant(0.1,tf.float32,[L]))

w3 = tf.Variable(tf.truncated_normal([4,4,L,M],stddev=0.1))
b3 = tf.Variable(tf.constant(0.1,tf.float32,[M]))

w4 = tf.Variable(tf.truncated_normal([7*7*M,N],stddev=0.1))
b4 = tf.Variable(tf.constant(0.1,tf.float32,[N]))

w5 = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
b5 = tf.Variable(tf.constant(0.1,tf.float32,[10]))


# In[6]:


# variable of probabilities calculated from Softmax
# reshape used to stretch x into one singe vector of 784 elements
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides = [1,2,2,1], padding = 'SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides = [1,2,2,1], padding = 'SAME') + b3)
yy3 = tf.reshape(y3, shape = [-1, 7 * 7 * M])
y4 = tf.nn.relu(tf.matmul(yy3, w4) + b4)
yy4 = tf.nn.dropout(y4, pkeep)
Ylogits = tf.matmul(yy4, w5) + b5
y = tf.nn.softmax(Ylogits)


# To calculate the error/loss in probabilities
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

is_correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))


# In[7]:


# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)


# In[8]:


for i in range(10000):
    # Load batch of images and correct answers
    batch_x,batch_y = mnist.train.next_batch(100)
    train_data = {x:batch_x, y_:batch_y}
    print(i)
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    
    # Start training
    session.run(train_step, {x:batch_x, y_: batch_y, lr: learning_rate, pkeep: 0.75})

np.savetxt('test1.txt', session.run(w1).flatten())
np.savetxt('bias1.txt', session.run(b1).flatten())
np.savetxt('test2.txt', session.run(w2).flatten())
np.savetxt('bias2.txt', session.run(b2).flatten())
np.savetxt('test3.txt', session.run(w3).flatten())
np.savetxt('bias3.txt', session.run(b3).flatten())
np.savetxt('test4.txt', session.run(w4).flatten())
np.savetxt('bias4.txt', session.run(b4).flatten())
np.savetxt('test5.txt', session.run(w5).flatten())
np.savetxt('bias5.txt', session.run(b5).flatten())

# first_image = mnist.test.images[0]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
# print(first_image)

