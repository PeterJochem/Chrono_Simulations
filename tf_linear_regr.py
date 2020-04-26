import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create a linear data set with some noise
x_batch = np.linspace(0, 100, 100)
y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 5.0 + 0.5

# Define the computational graph
x = tf.placeholder(tf.float32, shape=(None, ), name='x')
y = tf.placeholder(tf.float32, shape=(None, ), name='y')

w = tf.Variable(np.random.normal(), name='W')
b = tf.Variable(np.random.normal(), name='b')
		
y_pred = tf.add(tf.multiply(w, x), b)

loss = tf.reduce_mean(tf.square(y_pred - y))

# Calculate the output of the hidden layer
# hidden_out = tf.nn.relu(hidden)

# now let's define the cost function which we are going to train the model on
# Mean squared error

d_loss_dx = tf.gradients(loss, x)[0]

print("")
print("")

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_op = optimizer.minimize(loss)

epochs = 10
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
         
    feed_dict = {x: x_batch, y: y_batch }
		
    for i in range(epochs):
        sess.run(train_op, feed_dict)
        print(i, "loss:", loss.eval(feed_dict) )

    x_test = np.linspace(0, 100, 100)

    y_test = sess.run(y_pred, {x : x_batch})
    
    # grad = sess.run(d_loss_dx, {x : [1.0], y: [1.5] } )  

    print("")
    print("The gradient of loss wrt input")
    print(grad)

    plt.plot( x_test, y_test, color = "red" )
    plt.plot( x_batch, y_batch  )
    plt.ylabel('Prediction' )
    plt.show()





