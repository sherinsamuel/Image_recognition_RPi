
# coding: utf-8

# In[122]:


import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt 
np.set_printoptions(threshold=np.nan)

K = 6
L = 12
M = 24
N = 200

# To load weight and bias from files
wt1 = np.loadtxt('test1.txt').reshape(6,6,1,K)
bs1=np.loadtxt('bias1.txt').reshape(K,)
wt2 = np.loadtxt('test2.txt').reshape(5,5,K,L)
bs2=np.loadtxt('bias2.txt').reshape(L,)
wt3 = np.loadtxt('test3.txt').reshape(4,4,L,M)
bs3=np.loadtxt('bias3.txt').reshape(M,)
wt4 = np.loadtxt('test4.txt').reshape(7*7*M,N)
bs4=np.loadtxt('bias4.txt').reshape(N,)
wt5 = np.loadtxt('test5.txt').reshape(N,10)
bs5=np.loadtxt('bias5.txt').reshape(10,)

# to read image, resize, and flaten into  vector
img = Image.open('k6a.jpg')
img = img.convert('L')
img = img.point(lambda x: 0 if x>90 else 255, '1')
img = img.resize((28, 28), PIL.Image.ANTIALIAS)
img.save('resized_image.jpg')

in_data = imageio.imread('resized_image.jpg')
input_data = np.array(in_data).reshape(1,28,28,1)


# In[123]:


x = tf.placeholder(tf.float32,[1,28,28,1])

# Placeholder for all known probabilities of classification
y_ = tf.placeholder(tf.float32, [1,10])

# Variable to store values decided by TF
w1_ = tf.Variable(tf.truncated_normal([6,6,1,K],stddev=0.1))
b1_ = tf.Variable(tf.constant(0.1,tf.float32,[K]))
w1 = w1_.assign(wt1)
b1 = b1_.assign(bs1)

w2_ = tf.Variable(tf.truncated_normal([5,5,K,L],stddev=0.1))
b2_ = tf.Variable(tf.constant(0.1,tf.float32,[L]))
w2 = w2_.assign(wt2)
b2 = b2_.assign(bs2)

w3_ = tf.Variable(tf.truncated_normal([4,4,L,M],stddev=0.1))
b3_ = tf.Variable(tf.constant(0.1,tf.float32,[M]))
w3 = w3_.assign(wt3)
b3 = b3_.assign(bs3)

w4_ = tf.Variable(tf.truncated_normal([7*7*M,N],stddev=0.1))
b4_ = tf.Variable(tf.constant(0.1,tf.float32,[N]))
w4 = w4_.assign(wt4)
b4 = b4_.assign(bs4)

w5_ = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
b5_ = tf.Variable(tf.constant(0.1,tf.float32,[10]))
w5 = w5_.assign(wt5)
b5 = b5_.assign(bs5)

init = tf.global_variables_initializer()


# In[124]:


# variable of probabilities calculated from Softmax
# reshape used to stretch x into one singe vector of 784 elements
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides = [1,2,2,1], padding = 'SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides = [1,2,2,1], padding = 'SAME') + b3)
yy3 = tf.reshape(y3, shape = [-1, 7 * 7 * M])
y4 = tf.nn.relu(tf.matmul(yy3, w4) + b4)
Ylogits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(Ylogits)


# In[125]:


# init
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

test_data = {x:input_data}
print(session.run(y, feed_dict=test_data))
print(session.run(tf.argmax(y, 1), feed_dict=test_data))

