import numpy as np
import time
import random
import tensorflow as tf
from tensorflow import keras
from keras import backend as k
import matplotlib.pyplot as plt


import theano
import theano.tensor as T
import keras.backend as K

# Describe this class here
class trainInstance:

    # Constructor
    def __init__(self, inputData, label):
        
        self.input = inputData
        self.output = label


# This describes the entire dataset
class dataSet:
    
    # Constructor
    def __init__(self):
        
        self.allInstances = self.readInData()

    # Describe this method here
    def format(self, instances):
        length = len(instances) 
        widthInput = len(instances[0].input)
        widthOutput = len(instances[0].output)

        # Create zero matrices that we can then fill in
        allInputs = np.zeros( (length, widthInput) )
        allLabels = np.zeros( (length, widthOutput) )

        for i in range(len(instances) ):
            allInputs[i] = instances[i].input
            allLabels[i] = instances[i].output
        
        return allInputs, allLabels

    # Describe this method
    def printAll(self):

        for i in range( 100 ):
            print(self.allInstances[i].input)

    
    # This reads in the data and gets it ready to process
    # Input: selfi
    # Returns a list of all the input/output pairs in objects
    # of type trainInstance
    def readInData(self):
        
        # Open the file
        myFile = open("trainingData/trainingData.txt", "r")

        # List of all data
        allData = []
        
        # Read in the training data from the txt file
        line = myFile.readline()
        lineCount = 1

        while line:
            priorLine = line.split(" ")

            nextInput = np.zeros(3)
            lineCount = lineCount + 1
            # Get rid of spaces in list
            index = 0
            for i in range(len(priorLine) ):

                if (priorLine[index] == "0\n"):
                    nextInput[index] = 0.0
                    index = index + 1

                elif( priorLine[i] != ""):
                    nextInput[index] = float(priorLine[i])
                    index = index + 1
        
            label = np.zeros(3)

            line = myFile.readline()
            rawLabel = line
            rawLabel = rawLabel.split(" ")

            # Get rid of spaces in list
            index = 0
            for i in range(len(rawLabel) ):

                if (rawLabel[index] == "0\n"):
                    label[index] = 0.0
                    index = index + 1

                elif( rawLabel[i] != ""):
                    label[index] = float(rawLabel[i])
                    index = index + 1

            # def __init__(self, inputData, label):
            allData.append(trainInstance(nextInput, label) )
            
        print("")
        print("")
        print("There are " + str(lineCount) + " instances in this training set")
        print("")
        print("")
                
        return allData


# Create a datset object with all our data
myDataSet = dataSet()


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

loss = tf.nn.l2_loss(y_pred - y) # / (len(myDataSet.allInstances))
# loss = tf.reduce_mean(tf.square(y_pred - y))

d_loss_dx = tf.gradients(loss, x)[0]

print("")
print("")

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_op = optimizer.minimize(loss)


# Split the dataset into training and test sets
# np.random.shuffle(self.allInstances) 
trainSet = myDataSet.allInstances[ : int(0.80 * len(myDataSet.allInstances) ) ]
# print(trainSet)
# print(len(self.allInstances))
testSet = myDataSet.allInstances[int(0.80 * len(myDataSet.allInstances) ):  ]

# Format the dataset so Keras can use it
trainInputs, trainLabels = myDataSet.format(trainSet)
testInputs, testLabels = myDataSet.format(testSet)

# Train on the training set
epochs = 100
with tf.Session() as sess:
    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    feed_dict_train = {x: trainInputs, y: trainLabels }
    feed_dict_test = {x: testInputs, y: testLabels }

    loss_train_array = np.zeros(epochs)
    loss_test_array = np.zeros(epochs)  

    for i in range(epochs):
        sess.run(train_op, feed_dict_train)
        loss_train_array[i] = loss.eval(feed_dict_train)
        
        loss_test_array[i] = loss.eval(feed_dict_test)

    # Feed in a real pair of data we trained on
    grad_numerical = sess.run(d_loss_dx, {x: [testInputs[0] ] , y: [testLabels[0]] } )  
    print("")
    print("The gradient of loss wrt an input vector ")
    print(grad_numerical)

    # Feed in a random data point which doesn't belong to part of the dataset fucntion
    grad_numerical = sess.run(d_loss_dx, {x: [[1.0, 4.0, 8.0]] , y: [[4.0, 500.0, -20.0]] } )
    print("")
    print("The gradient of loss wrt a random input vector (ie not from our observed dataset)")
    print(grad_numerical)



plt.plot( np.linspace(0, epochs, epochs), loss_test_array, color = "blue" )
plt.title( 'Loss Function vs Epoch - Chrono Ball Bouncing Dataset' )
plt.ylabel( 'Loss' )
plt.xlabel( 'Epoch' )
plt.show()



