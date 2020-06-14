import numpy as np
import time
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore') # Gets rid of future warnings
with warnings.catch_warnings():
    import tensorflow as tf
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Globals for setting the depth of the predictions 
maxDepth = -100.0
minDepth = 1000.0


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
        
        # FIX ME - replace this with read in all files data
        # belta0  belta30  belta-30  belta45  belta-45  belta60  belta-60  belta90 
        self.allInstances = self.readInFile("/extract/belta0/")
        self.allInstances.extend(self.readInFile("/extract/belta30/"))
        self.allInstances.extend(self.readInFile("/extract/belta-30/"))
        self.allInstances.extend(self.readInFile("/extract/belta45/"))
        self.allInstances.extend(self.readInFile("/extract/belta-45/"))
        self.allInstances.extend(self.readInFile("/extract/belta60/"))
        self.allInstances.extend(self.readInFile("/extract/belta-60/"))
        self.allInstances.extend(self.readInFile("/extract/belta90/"))        
        
        # beta_0  beta_30  beta_-30  beta_45  beta_-45  beta_60  beta_-60  beta_90
        self.allInstances.extend(self.readInFile("/intrude/beta_0/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_30/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_-30/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_45/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_-45/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_60/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_-60/"))
        self.allInstances.extend(self.readInFile("/intrude/beta_90/"))
    
        #######
        # Fix me - add the small angles files
        #######


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
    def readInFile(self, folderName):
        
        global minDepth
        global maxDepth

        # Open the file
        basePrefix = "/home/peter/Desktop/Chrono/chrono/JuntaoData/Foot-ROBOT/data_set/"
        inputFilePath = str(basePrefix) + str(folderName) + str("output_plate_positions.csv") 
        labelFilePath = str(basePrefix) + str(folderName) + str("output_plate_forces.csv")
        myInputFile = open(inputFilePath)
        
        myLabelFile = open(labelFilePath)
        
        # Describe the angles of the foot
        gamma = -1.0
        beta = -1.0

        # List of all data
        allData = []
        
        # Read in the training data from the txt file
        line = myInputFile.readline()
        
        lineCount = 0

        while line:
            
            line = line.split(",")

            if (len(line) < 2):
                print("Error: Line length is less than 2")
                
                # Read in the next line
                line = myInputFile.readline()
                rawLabel = myLabelFile.readline()
                rawLabel = rawLabel.split(",")


            if ( (len(line) == 2) or (line[2] == '') ):
                # print("New gamma, beta pairs")
                # Set gamma and beta 
                gamma = line[0]
                beta = line[1]

                # Read in the next line
                line = myInputFile.readline()
                rawLabel = myLabelFile.readline()
                rawLabel = rawLabel.split(",")
                continue

             
            # The FILE's data is time, x, y, z, depth ...
            # The data we care about is ForceX/depth, Force_Y/depth  
            nextInput = np.zeros(3)
            lineCount = lineCount + 1
            
            # Get rid of spaces in list
            index = 0
            
            label = np.zeros(2)

            # The label file is time, F1, F2, F3
            rawLabel = myLabelFile.readline()
            rawLabel = rawLabel.split(",")
    

            # Ignore data if it isn't a full line - happens when I ctrl c on Chrono
            if ( (len(line) >= 5) ):

                try:
                    forceX = float( rawLabel[1] )    
                    forceZ = float( rawLabel[3] )
                    depth = float( line[4] ) 
                    
                    if (depth < minDepth):
                        minDepth = depth
                    #    print(1)
                    if (depth > maxDepth):
                        maxDepth = depth
                    #    print(2)
                    # else:
                    #    print(depth)
                    
                    nextInput[0] = gamma  
                    nextInput[1] = beta
                    # I added depth as another input to the function
                    nextInput[2] = depth 

                    label[0] = float( float(forceX) / depth) 
                    label[1] = float( float(forceZ) / depth)
                except:
                    pass 

            allData.append(trainInstance(nextInput, label) )
            
            # Read in the next line
            line = myInputFile.readline()
                
        return allData

# Describe this method
def createGraphs(depth, sess):
    angleResolution = 20
    # Remember we are using DEGREES
    increment = 180.0 / angleResolution

    inputData = []

    F_X = np.zeros((angleResolution, angleResolution))

    F_Z = np.zeros((angleResolution, angleResolution))

    count = 0
    for i in range(angleResolution):
        for j in range(angleResolution):

            gamma = -1 * (180.0 / 2.0) + (j * increment)
            beta = (180.0 / 2.0) - (i * increment)

            nextEntry = [gamma, beta, depth]

            inputData.extend( [nextEntry] )
            count = count + 1

            # The y-label is irrelevant to predicting the y-label, here, after training
            prediction = sess.run(y_pred, {x: [nextEntry] , y: [[ 1.0, 1.0 ]] } )

            # F_X[i][j] = ( (prediction[0][0] + 0.1) / 0.2) * 255
            # F_Z[i][j] = ( (prediction[0][1] + 0.3) / 0.6) * 255

            F_X[i][j] = prediction[0][0]
            F_Z[i][j] = prediction[0][1]

    w = 10
    h = 10
    fig = plt.figure(figsize = (8, 8))
    columns = 3
    rows = 1
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.imshow(F_Z)
    ax1.set_ylabel('β')
    ax1.set_xlabel('γ')
    ax1.title.set_text('α_Z at depth: ' + str(depth) )
    plt.colorbar()

    ax2 = fig.add_subplot(rows, columns, 3)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    plt.imshow(F_X)
    ax2.set_ylabel('β')
    ax2.set_xlabel('γ')
    ax2.title.set_text('α_X at depth: ' + str(depth) )

    plt.colorbar()
    plt.show()
    fig.savefig( str("graphs/") + str(depth) + str('.png'))



# Create a datset object with all our data
myDataSet = dataSet()

# Define the computational graph
x = tf.placeholder(tf.float32, shape=(None, 3), name = 'x')

# Outputs are [F x / depth, Force z / depth]
y = tf.placeholder(tf.float32, shape=(None, 2), name = 'y')

W1 = tf.Variable(tf.random_normal([3, 20], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.random_normal([20]), name = 'b1')

W2 = tf.Variable(tf.random_normal([20, 2], stddev = 0.03), name = 'W2')
b2 = tf.Variable(tf.random_normal([2]), name = 'b2')

# W3 = tf.Variable(tf.random_normal([70, 40], stddev = 0.03), name = 'W3')
# b3 = tf.Variable(tf.random_normal([40]), name = 'b3')

# and the weights connecting the hidden layer to the output layer
# W4 = tf.Variable(tf.random_normal([40, 15], stddev = 0.03), name = 'W4')
# b4 = tf.Variable(tf.random_normal([15]), name = 'b4')

# and the weights connecting the hidden layer to the output layer
# W5 = tf.Variable(tf.random_normal([15, 2], stddev = 0.03), name = 'W5')
# b5 = tf.Variable(tf.random_normal([2]), name = 'b5')


hidden_out1 = tf.add(tf.matmul(x, W1), b1)
hidden_out1 = tf.nn.relu(hidden_out1)

#hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
#hidden_out2 = tf.nn.relu(hidden_out2)

#hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)
#hidden_out3 = tf.nn.relu(hidden_out3)

#hidden_out4 = tf.add(tf.matmul(hidden_out3, W4), b4)
#hidden_out4 = tf.nn.relu(hidden_out4)

y_pred = tf.add(tf.matmul(hidden_out1, W2), b2)

# loss = tf.nn.l2_loss(y_pred - y) # / (len(myDataSet.allInstances))
loss = tf.reduce_mean(tf.square(y_pred - y))

d_loss_dx = tf.gradients(loss, x)[0]


# 0.00000001
optimizer = tf.train.GradientDescentOptimizer(0.001)
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

# Train on the training set
epochs = 4000
with tf.Session() as sess:

    # Initialize the variables
    sess.run(tf.global_variables_initializer())
        
    save_path = saver.save(sess, "myNetworks/myNet.ckpt" )

    feed_dict_train = {x: trainInputs, y: trainLabels }
    feed_dict_test = {x: testInputs, y: testLabels }

    loss_train_array = np.zeros(epochs)
    loss_test_array = np.zeros(epochs)  

    for i in range(epochs):
        sess.run(train_op, feed_dict_train)
        loss_train_array[i] = loss.eval(feed_dict_train)
        
        loss_test_array[i] = loss.eval(feed_dict_test)

    # Feed in a real pair of data we trained on
    # grad_numerical = sess.run(d_loss_dx, {x: [testInputs[0] ] , y: [testLabels[0] ] } )  
    prediction = sess.run(y_pred, {x: [testInputs[0] ] , y: [testLabels[0] ] } )
    print("")
    print("The prediction is ")
    print(prediction)


    plt.plot( np.linspace(0, epochs, epochs), loss_test_array, color = "blue" )
    plt.title( 'Loss Function vs Epoch - Hopping Foot' )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epoch' )
    plt.show()

    # Loop over the possible range of depths
    # from min depth to max depth

    numGraphs = 5
    increment = float((maxDepth - minDepth)) / numGraphs

    for i in range(numGraphs):
        depth = minDepth + (i * increment)
        createGraphs(depth, sess)
        

    """
    angleResolution = 20
    depth = 0.1
    # Remember we are using DEGREES
    increment = 180.0 / angleResolution
    
    inputData = []
    
    F_X = np.zeros((angleResolution, angleResolution))
    
    F_Z = np.zeros((angleResolution, angleResolution))
    
    depth = 8

    count = 0
    for i in range(angleResolution):
        for j in range(angleResolution):
            
            gamma = -1 * (180.0 / 2.0) + (j * increment) 
            beta = (180.0 / 2.0) - (i * increment)
            
            nextEntry = [gamma, beta, depth]
            
            inputData.extend( [nextEntry] )
            count = count + 1
                
            # The y-label is irrelevant to predicting the y-label, here, after training
            prediction = sess.run(y_pred, {x: [nextEntry] , y: [[ 1.0, 1.0 ]] } )
            
            # F_X[i][j] = ( (prediction[0][0] + 0.1) / 0.2) * 255
            # F_Z[i][j] = ( (prediction[0][1] + 0.3) / 0.6) * 255

            F_X[i][j] = prediction[0][0] 
            F_Z[i][j] = prediction[0][1]

    print("The min-depth is " + str(minDepth) )
    print("The max depth is " + str(maxDepth) )

    w = 10
    h = 10
    fig = plt.figure(figsize = (8, 8))
    columns = 3
    rows = 1
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    plt.imshow(F_Z)
    ax1.set_ylabel('β')
    ax1.set_xlabel('γ')
    ax1.title.set_text('α_Z')
    plt.colorbar()

    ax2 = fig.add_subplot(rows, columns, 3)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    plt.imshow(F_X)
    ax2.set_ylabel('β')
    ax2.set_xlabel('γ')
    ax2.title.set_text('α_X')
    
    plt.colorbar()
    plt.show()
    """
    
    
