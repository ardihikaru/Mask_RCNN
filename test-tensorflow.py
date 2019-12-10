# example from https://learningtensorflow.com/lesson10/

import sys
import numpy as np

# Disable WARNING:
import os
# The TensorFlow library wasn't compiled to use SSE4.1 instructions,
# but these are available on your machine and could speed up CPU computations.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from datetime import datetime

print(" >>>> Tensorflow VERSION = ", tf.__version__)

"""
available devices are [ 
    /job:localhost/replica:0/task:0/device:CPU:0, 
    /job:localhost/replica:0/task:0/device:XLA_CPU:0, 
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 
]
"""

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/job:localhost/replica:0/task:0/device:XLA_GPU:0"
    # device_name = "/gpu:0"
else:
    # device_name = "/job:localhost/replica:0/task:0/device:CPU:0"
    device_name = "/job:localhost/replica:0/task:0/device:XLA_CPU:0"
    # device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)

# with tf.device(device_name):
#     try:
#         # random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
#         random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
#     except:
#         random_matrix = tf.keras.backend.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
# with tf.compat.v1.ConfigProto() as session:
    result = session.run(sum_operation)
    print(result)
# try:
#     # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#     with tf.compat.v1.ConfigProto(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)
# except:
#     with tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print(" >>>>> Shape:", shape, "Device:", device_name)
print(" >>>>> Time taken:", datetime.now() - startTime)

print("\n" * 5)
