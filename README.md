# Description
I spent this week learning about how to call Python code from Matlab. I put together a rough sketch of what the final API might look like. It lets Matlab use a tensorflow neural network. It also lets Matlab get the derivates of the output variables with respect to the input variables. I also spent time learning how to use Juntao's simulation of the foot contacting the granular material.  

# Tensorflow and Automatic Differentiation 
Tensorflow and most other (if not all) machine learning packages use automatic differentiation to compute derivatives. One could use numerical differentiation but for neural networks this is not practical. Gradient descent methods require the partial derivative of each parameter in the model with respect to the loss. The network could have billions of weights and numerical derivatives would require use to forward propagate in proportion to the number of weights. Numerical derivatives often suffer from more numerical error than other alternatives.

Automatic differentiation computes derivatives of a function by first decomposing the function into many elemental pieces, finding the derivative of each piece, and then using the chain rule to find the overall derivatives (or partials). An example would be a linear regression model with two parameters, lets call them W and b. Lets also call Y the label and X our input. Then, Loss =  Y - max(0, W*X + b). We can decompose this function into the following computational graph. 

![alt text](https://miro.medium.com/max/726/1*W6-39saZm_QqL-wQvGESGQ.png "Computational Graph")

The insight of the famous backprop algorithm is that a lot of this work is reduandant.

We can compute the partial deivatives of weights closer to the end of the network and "back propagate" those values to earlier layers where the partial derivs depend on those later derivatives. This lets us avoid recomputing the same derivatives over and over. But... we need an algorithmic way to automate this process.

A really good resource on automatic differentiation: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf



# Persistent Storage of Data?
Since it requires a lot of time to run granular simulations, it probably makes sense to store the data in database to avoid resimulating any data. We could probably get away with just using .txt files but its probably worth the effort to use a real database. It will be easier to understand later and modify etc.


# Questions



# Next Week



# Notes to Self
