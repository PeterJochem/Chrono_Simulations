# Description
I spent time learning how to use Juntao's simulation of the foot contacting the granular material. I then used the simulation to generate a dataset of pairs of (x, x_dt, y, y_dt, z, z_dt) and the GRF (F_x, F_y, F_z). I naively ran the initial dataset with the neural network and it learned a good representation of the function. I then looked at the csv files and most of the data is a non-collision state. I made some changes to the simulation code and created a small dataset of instances mostly when the foot and granular material are in contact. An image of the results is below. I need to spend time this week generating more training data. The dataset I used to create the graph was from about 20 minutes of time on the gpu so its still a relatively small dataset.  

I also spent time this week learning about how to call Python code from Matlab. I put together a demo of what the final Matlab API might look like. It lets Matlab use a Python Tensorflow neural network. 
![alt text](https://github.com/PeterJochem/Chrono_Simulations/blob/master/HoppingNeural.png "Granular Foot GRF")

Minimized the average loss per training example across the entire dataset with gradient descent

# Tensorflow and Automatic Differentiation 
Tensorflow and most other (if not all) machine learning packages use automatic differentiation to compute derivatives. One could use numerical differentiation but for neural networks this is not practical. Gradient descent methods require the partial derivative of each parameter in the model with respect to the loss. The network could have billions of weights and numerical derivatives would require use to forward propagate in proportion to the number of weights. Numerical derivatives also suffer from more numerical error than automatic differentiation. 

Automatic differentiation computes derivatives of a function by first decomposing the function into many elemental pieces, finding the derivative of each piece, and then using the chain rule to find the overall derivative (or partials). An example would be a linear regression model with two parameters, lets call them W and b. Lets also call Y the label and X our input. Then, Loss =  Y - max(0, W*X + b). We can decompose this function into the following computational graph. 

![alt text](https://miro.medium.com/max/726/1*W6-39saZm_QqL-wQvGESGQ.png "Computational Graph")


# Implementing a Model in Tensorflow
How does one implement a model in Tensorflow? There are a few steps. First, we specify the computational graph using the *.tf library functions. We specify nodes and operations between them. We then initiliaze the weights randomnly. Finally, we evaluate the loss function for known training data. Over the set of all labeled training data, we can use automatic differentiation to compute the gradient of loss wrt the weights. Once we know the gradient, we can use one of Tensorflow's many gradient based optimization algorithms to reduce the loss on the training data.


### Resources 
A really good resource on automatic differentiation: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf


# Persistent Storage of Data?
Since it requires a lot of time to run a granular simulations (about 1.5 hours on gpu), it probably makes sense to store the data in a database to avoid resimulating any data. We could probably get away with just using .txt files but its probably worth the effort to use a real database. It will be easier to understand later and modify etc.


# Questions
How to validate the model's gradient calculations? Is it possible for the model to have a high accuracy but the derivatives of the model be poor? I am guessing one could engineer a case where this is true but its probably unlikely to occur on its own? Its actually an interesting question on its own and I am curious to know. If this is not generally possible/likely to occur, we can probably assume that if the model has good predictive accuracy then it's derivatives will be relatively correct as well.

# Next Week
Fine tune the Matlab-Python interface. Generate dataset with more collisions with granular material. Build simple database to manage the data. 

# Notes to Self
