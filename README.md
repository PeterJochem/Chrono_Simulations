# Description
I spent time learning how to use Juntao's simulation of the foot contacting the granular material. I then used the simulation to generate a dataset of pairs of (x, x_dt, y, y_dt, z, z_dt) and the GRF (F_x, F_y, F_z).
I also spent time this week learning about how to call Python code from Matlab. I put together a demo of what the final Matlab API might look like. It lets Matlab use a Python Tensorflow neural network. It also lets Matlab get the derivates of the output variables with respect to the input variables. 

# Tensorflow and Automatic Differentiation 
Tensorflow and most other (if not all) machine learning packages use automatic differentiation to compute derivatives. One could use numerical differentiation but for neural networks this is not practical. Gradient descent methods require the partial derivative of each parameter in the model with respect to the loss. The network could have billions of weights and numerical derivatives would require use to forward propagate in proportion to the number of weights. Numerical derivatives often suffer from more numerical error than other alternatives.

Automatic differentiation computes derivatives of a function by first decomposing the function into many elemental pieces, finding the derivative of each piece, and then using the chain rule to find the overall derivatives (or partials). An example would be a linear regression model with two parameters, lets call them W and b. Lets also call Y the label and X our input. Then, Loss =  Y - max(0, W*X + b). We can decompose this function into the following computational graph. 

![alt text](https://miro.medium.com/max/726/1*W6-39saZm_QqL-wQvGESGQ.png "Computational Graph")


An added bonus of automatic differentiation is that alot of the d_loss / d_weight_i's depend on each other. We can store the values and reuse them. In other words, the big insight of the famous backprop algorithm is that a lot of this work is reduandant. Automatic differentiation is a convenient way to store partial results we can use later. We can compute the partial deivatives of weights closer to the end of the network and "back propagate" those values to earlier layers where the partial derivs depend on those later derivatives. This lets us avoid recomputing the same partial derivatives over and over.

How does one implement a model in Tensorflow? There are a few steps. First, we specify the computational graph using the *.tf library functions. We specify nodes and operations between them. We then initiliaze the weights randomnly. Finally, we evaluate the loss function for known training data. Over the set of all labeled training data, we can use automatic differentiation to compute the gradient of loss wrt the weights. Once we know the gradient, we can use one of Tensorflow's many gradient based optimization algorithms to reduce the loss on the training data.


### Resources 
A really good resource on automatic differentiation: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf


# Persistent Storage of Data?
Since it requires a lot of time to run a granular simulations (about 1.5 hours on gpu), it probably makes sense to store the data in database to avoid resimulating any data. We could probably get away with just using .txt files but its probably worth the effort to use a real database. It will be easier to understand later and modify etc.


# Questions

# Next Week

# Notes to Self
