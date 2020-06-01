import numpy as np
import time
import random
import sys
import warnings

warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import tensorflow as tf
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# from tensorflow import keras
# from keras import backend as k
# import matplotlib.pyplot as plt

#import os
#os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

# Describe this class here
class trainInstance:

    # Constructor
    def __init__(self, inputData, label):
        
        self.input = inputData
        self.output = label


# This describes the entire dataset
class dataSet:
    
    # Constructor
    def __init__(self, readData):
        
        if (readData == True):
            self.allInstances = self.readInData()
        else:
            pass

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
            
        print("Reading in the data")

        # Open the file
        # myFile = open("trainingData/trainingData.txt", "r")
        myInputFile = open("output_plate_positions.csv")
        
        myLabelFile = open("output_plate_forces.csv")

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


# Create a datset object with all our data and don't read in the dataset
myDataSet = dataSet(False)


# Define the computational graph
# Inputs are ξ f = 
x = tf.placeholder(tf.float32, shape=(None, 5), name = 'x')

# Outputs are [F x , F y , M z]
y = tf.placeholder(tf.float32, shape=(None, 3), name = 'y')

W1 = tf.Variable(tf.random_normal([4, 30], stddev = 0.03), name = 'W1')
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


# Setup the saving of the network
saver = tf.train.Saver()

# Describe this function
# Describe the network
def predict(V1, V2, V3):
      
    pred_grf = 0
    with tf.Session() as sess:
    
        # No need to re-nitialize the variables
        # sess.run(tf.global_variables_initializer())
        # Load the weights from the saved files
        saver.restore(sess, "myNetworks/myNet.ckpt")

        # Feed in a real pair of data we trained on
        # Note to self - the y label does not matter for forward propping values
        pred_grf = sess.run(y_pred, {x: [[ V1, V2, V3 ]] , y: [ [00.0, 20.0, 500.0] ] } )  
        print("")
        print("")
        print("The prediction is" + str(pred_grf) )
        print("")
        print("")
        
    return pred_grf

print("")
sys.stdout.write(str(predict(0, 0, 6.64294) ) )
print()
print("")
# computeGRF()


