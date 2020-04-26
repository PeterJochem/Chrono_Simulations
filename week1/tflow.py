import numpy as np
import tensorflow as tf


tf.enable_eager_execution()

x = tf.ones((2, 2)) * 4 

print(x)

with tf.GradientTape() as t:
  t.watch(x)
  # y = tf.reduce_sum(x)
  y = x
  z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in range(2):
  for j in range(2):
    # assert dz_dx[i][j].numpy() == 8.0
        
    print(dz_dx[i][j].numpy())

print("")
print("")
print("Hello World")
print("")
print("")








