import numpy as np
import time
import random
import tensorflow as tf
from tensorflow import keras
from keras import backend as k

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

    # Describe this method
    def train(self, network):
        
        # Split the dataset into training and test sets
        # np.random.shuffle(self.allInstances) 
        trainSet = self.allInstances[ : int(0.80 * len(self.allInstances) ) ]
        # print(trainSet)
        # print(len(self.allInstances))
        testSet = self.allInstances[int(0.80 * len(self.allInstances) ):  ]

        # Format the dataset so Keras can use it
        trainInputs, trainLabels = self.format(trainSet)
        testInputs, testLabels = self.format(testSet)
        
        # Do some training on the training set - use "verbose = 0" to not print epoch info
        network.fit( trainInputs, trainLabels, epochs = 10, verbose = 5)
                    
        # Compute the error on the test set
         
        # self.printAll()
        print(len(self.allInstances) )


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


# Define the neural network structure
network = keras.Sequential([
    keras.layers.Dense(30, input_dim = 3, activation = 'tanh'),
    keras.layers.Dense(20, activation ='tanh'),
    keras.layers.Dense(10, activation ='tanh'),
    keras.layers.Dense(3)
])


# Set more of the model's parameters
# What does this parameter do?
optimizer = tf.keras.optimizers.RMSprop(0.00005)

network.compile(loss='mse',
                optimizer = optimizer,
                metrics = ['mae', 'mse'])

# Create a datset object with all our data
myDataSet = dataSet()

# Train the network
myDataSet.train(network)

# Make predictions about the data
print( network.predict( np.array( [[0.0, 0.7, 0.0]] ) ) )

outputTensor = network.output

listOfVariableTensors = network.trainable_weights

gradients = k.gradients(outputTensor, listOfVariableTensors)

# print(gradients)

trainingExample = np.array( [[ 0.0, 0.80, 0.0]]  )
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
evaluated_gradients = sess.run(gradients,feed_dict={network.input:trainingExample})

print(evaluated_gradients)
