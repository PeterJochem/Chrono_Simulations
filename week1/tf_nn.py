import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

# Create a linear data set with some noise
# x_batch = np.linspace(0, 10, 100)
# y_batch = 1.5 * (x_batch * x_batch) + np.random.randn(*x_batch.shape) * 5.0 + 0.5

x_batch = np.zeros((200, 3))
for i in range(len(x_batch) ):
    for j in range(3):
        x_batch[i][j] = i

y_batch = copy.deepcopy(x_batch) * 2

# Define the computational graph
# Inputs are ξ f = [x f , ẋ f , y f , ẏ f , θ f , θ̇ f]
# The ball dataset is diffrent 
x = tf.placeholder(tf.float32, shape=(None, 3), name = 'x')

# Outputs are [F x , F y , M z]
y = tf.placeholder(tf.float32, shape=(None, 3), name = 'y')

W1 = tf.Variable(tf.random_normal([3, 30], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([30]), name ='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([30, 3], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([3]), name = 'b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_pred = tf.add(tf.matmul(hidden_out, W2), b2)

loss = tf.nn.l2_loss(y_pred - y) / (len(x_batch))
# loss = tf.reduce_mean(tf.square(y_pred - y))

d_loss_dx = tf.gradients(loss, x)[0]

print("")
print("")

optimizer = tf.train.GradientDescentOptimizer(0.000001)
train_op = optimizer.minimize(loss)

epochs = 100
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
         
    feed_dict = {x: x_batch, y: y_batch }
		
    loss_array = np.zeros(epochs)
    for i in range(epochs):
        sess.run(train_op, feed_dict)
        loss_array[i] = loss.eval(feed_dict)

    # x_test = np.linspace(0, 10, 100)
    # y_test = sess.run(y_pred, {x : x_batch})
    
    # grad = sess.run(d_loss_dx, {x : [1.0], y: [1.5] } )  

    # print("")
    # print("The gradient of loss wrt input")
    # print(grad)

    # plt.plot( x_test, y_test, color = "red" )
    # plt.plot( x_batch, y_batch, color = "green" )
    plt.plot( np.linspace(0, epochs, epochs), loss_array, color = "blue" )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epoch' )
    plt.show()





