# a tensor is a generalizetion of a scalor vector
# a vector is one dim tensor
# a matrix is a 2 dim tensor

# for error massages handeling
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# init tensors
# x = tf.constant(5)  # int32
# x = tf.constant(5.0)  # float32
# x = tf.constant((1, 1))  # int32
# x = tf.constant((1, 1), dtype=tf.float32)
# x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
# x = tf.ones((3, 3))
# x = tf.zeros((2,3))
# x = tf.eye((3))
# x = tf.random.normal((3, 3), mean=0, stddev=1)
# x = tf.random.uniform((1, 3), minval=0, maxval=1)
# x = tf.range(start=1, limit=10, delta=2)  # delta is the step

# x = tf.cast(x, dtype=tf.float64)  # cast to another data type

# print(x)
#
# with tf.compat.v1.Session() as sess:
#     print(x.eval())


# tensor math

# x = tf.constant([1, 2, 3])
# y = tf.constant([9, 8, 7])

# z = tf.add(x, y)
# z = x + y

# z = tf.subtract(x, y)
# z = x - y

# z = tf.divide(x, y)
# z = x / y

# z = tf.multiply(x, y)
# z = x * y

# z = tf.tensordot(x, y, axes=1)
# z = tf.reduce_sum(x*y, axis=0)

# z = x ** 5

# x = tf.random.normal((2, 3))
# y = tf.random.normal((3, 4))
# z = tf.matmul(x, y)
# z = x @ y

# x = tf.constant([0, 1, 0, 1, 1, 3, 3, 4, 1, 2, 3, 4, 0, 3])
# z = x[::2] # skip evety odd element
#

# indexing

# indecies = tf.constant([0])
# x_ind = tf.gather(x, indecies)


# x = tf.constant([[1, 2],
#                  [3, 4],
#                  [5, 6]]
#                 )
#
# print(x)


# reshaping
with tf.compat.v1.Session() as sess:
    x = tf.range(9)
    print(x.eval())
    x = tf.reshape(x, (3, 3))
    print(x.eval())
    x = tf.transpose(x, perm=[1, 0])
    print(x.eval())




