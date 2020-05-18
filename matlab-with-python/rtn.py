import numpy as np
import time
import random
import sys
import os

import warnings
warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    # from keras import backend as k
    import matplotlib.pyplot as plt
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



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
        # myFile = open("trainingData/trainingData.txt", "r")
        myInputFile = open("/home/peter/Desktop/Chrono/chrono/template_project/output_plate_positions_and_velocities.csv")
        
        myLabelFile = open("/home/peter/Desktop/Chrono/chrono/template_project/sim_data/output_plate_forces.csv")

        # List of all data
        allData = []
        
        # Read in the training data from the txt file
        line = myInputFile.readline()
        lineCount = 0

        while line:
            
            line = line.split(",")
            # The data is time, x, dx_dt, y, dy_dy, ...
            nextInput = np.zeros(6)
            lineCount = lineCount + 1
            # Get rid of spaces in list
            index = 0
            
            # Ignore data if it isn't a full line - happens when I ctrl c on Chrono
            if (len(line) >= 7):
                for i in range(len(line) ):

                    if ( i == 0 ):
                        # We want to ignore the time value
                        continue

                    elif (line[i] == "\n"):
                        continue

                    elif( line[i] != ""):
                        # print(line[i])
                        nextInput[index] = float(line[i])
                        index = index + 1
        
            # print(nextInput)

            label = np.zeros(3)

            # The label file is t, F1, F2, F3
            rawLabel = myLabelFile.readline()
            rawLabel = rawLabel.split(",")

            # Get rid of spaces in list
            index = 0

            # Ignore data if it isn't a full line - happens when I ctrl c on Chrono
            if (len(rawLabel) >= 3):
                for i in range(len(rawLabel) ):
                
                    if ( i == 0 ):
                        continue
                
                    if (rawLabel[i] == "\n"):
                        continue

                    elif( rawLabel[i] != ""):
                        label[index] = float(rawLabel[i])
                        index = index + 1

            # print(label)
            
            # def __init__(self, inputData, label):
            allData.append(trainInstance(nextInput, label) )
            
            # Read in the next line
            line = myInputFile.readline()

        

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
x = tf.placeholder(tf.float32, shape=(None, 6), name = 'x')

# Outputs are [F x , F y , M z]
y = tf.placeholder(tf.float32, shape=(None, 3), name = 'y')

W1 = tf.Variable(tf.random_normal([6, 30], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([30]), name ='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([30, 3], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([3]), name = 'b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_pred = tf.add(tf.matmul(hidden_out, W2), b2)

# loss = tf.nn.l2_loss(y_pred - y) # / (len(myDataSet.allInstances))
loss = tf.reduce_mean(tf.square(y_pred - y))

d_loss_dx = tf.gradients(loss, x)[0]

print("")
print("")

optimizer = tf.train.GradientDescentOptimizer(0.000001)
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

# Setup the saving of the network
saver = tf.train.Saver()

# Describe this function
# Describe the network
def computeGRF(V1, V2, V3, V4, V5, V6):
      
    pred_grf = 0
    with tf.Session() as sess:
    
        # No need to re-nitialize the variables
        # sess.run(tf.global_variables_initializer())
        # Load the weights from the saved files
        saver.restore(sess, "/home/peter/Desktop/Chrono/chrono/template_project/matlab-with-python/myNetworks/myNet.ckpt")

        # Feed in a real pair of data we trained on
        pred_grf = sess.run(y_pred, {x: [[ V1, V2, V3, V4, V5, V6 ]] , y: [ [100.0, 200.0, 500.0] ] } )  
        print("")
        print("")
        print(pred_grf)
        print("")
        print("")
        
    return pred_grf

sys.stdout.write(str(computeGRF(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) ) )
# computeGRF()


