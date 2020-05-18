# Description
This week I setup the Chrono simulation to use the trained neural network's predictions as the simulations GRF. I read more into the Chrono API. Juntao also helped me with this. After, I figured out the changes needed to the Chrono simulation code, I had to figure out how to embedd Python code in C++. Python provides an API to embedd Python in C/C++ (https://docs.python.org/3.7/extending/embedding.html).

I made the changes the the Chrono simulation to let the neural network compute the GRFs from C++. I then ran the simulation with Juntao's new granular JSON - letting Chrono compute GRF itself, trained the network on this new simulation data, and then ran a simulation letting the neural network compute the GRF at each time step.

Both scenarios have the same initial conditions so the neural network should have a high degree of success. The results are below     


![alt text](https://github.com/PeterJochem/Chrono_Simulations/blob/master/smallVis.png "0.5 Second Visualization")


# Next Week
1) Generate more data and train the network on a larger dataset

Compare neural network's predicted GRF with Dan's RFT-based GRFs?

