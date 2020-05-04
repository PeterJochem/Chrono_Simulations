import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create a linear data set with some noise
x_batch = np.linspace(0, 100, 100)
y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 5.0 + 0.5

# Define the computational graph
x = tf.placeholder(tf.float32, shape=(None, ), name = 'x') # The input to the function
y = tf.placeholder(tf.float32, shape=(None, ), name = 'y') # The associated label

# Learable parameters of the model
w = tf.Variable(np.random.normal(), name = 'W')
b = tf.Variable(np.random.normal(), name = 'b')

# Node that is the model's prediction
y_pred = tf.add(tf.multiply(w, x), b)

# Graph's node which is the loss
loss = tf.reduce_mean(tf.square(y_pred - y))

# Add gradient nodes to the computational graph
d_loss_dx = tf.gradients(loss, x)[0]
d_loss_dw = tf.gradients(loss, w)[0]

# Choose an optimization algorithm
optimizer = tf.train.GradientDescentOptimizer(0.0001)
# Specify which node in the graph to minimize
train_op = optimizer.minimize(loss)

# Run data and minimze loss
epochs = 10
with tf.Session() as sess:

    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # Tells model what the input/output pairs are
    feed_dict = {x: x_batch, y: y_batch }
		
    for i in range(epochs):

        # This specefies how what the input data is and how to
        # update the weights - use train_op object to update weights
        sess.run(train_op, feed_dict)

        # Print the loss at the current point with input data 
        # as the data in the feed_dictionary
        print(i, "loss:", loss.eval(feed_dict) )

 
    # This is a example of how to compute derivatives using the learned model
    grad = sess.run(d_loss_dw, {x : x_batch, y: y_batch } )

    print("")
    print("The gradient of loss wrt w is ")
    print(grad)
    
    # Generate trend line using the learned function
    x_test = np.linspace(0, 100, 100)
    y_test = sess.run(y_pred, {x : x_test})

    # Plot the data
    plt.plot( x_test, y_test, color = "red" )
    plt.plot( x_batch, y_batch  )
    plt.ylabel('Prediction' )
    plt.show()





