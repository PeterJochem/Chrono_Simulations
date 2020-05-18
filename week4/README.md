# Description
This week I setup the Chrono simulation to use the trained neural network's predictions as the simulations GRF. I read more into the Chrono API. Juntao also helped me with this. After I figured out the changes needed to the Chrono simulation code, I had to figure out how to embed Python code in C++. Python provides an API to embed Python in C/C++ (https://docs.python.org/3.7/extending/embedding.html).

I made the changes to the Chrono simulation to let the neural network compute the GRFs from C++. I then ran the simulation with Juntao's new granular JSON - letting Chrono compute GRF itself, trained the network on this new simulation data, and then ran a simulation letting the neural network compute the GRF at each time step.

Both scenarios have the same initial conditions so the neural network should have a high degree of success. The results are below     

The original motion
![alt text](https://github.com/PeterJochem/Chrono_Simulations/blob/master/originalMotion.png "")

The motion using the neural network's predicted GRF
![alt text](https://github.com/PeterJochem/Chrono_Simulations/blob/master/nn_pred_GRF_Motion.png "")

The qualitative behavior is the same. Although, the final position of the two motions is slightly diffrent. Something to remember is that the error at each time step should compound on itself. We have no correction mechanism. A slight error in the GRF prediction at each time step can impact all future timestep's predictions of the GRF. This shows that the neural network is being trained well and that the interface between Python and C++ is computing what I wanted it to. 


# Next Week
1) Generate more data, train the network on this larger dataset, and then see how the neural network makes predictions on new, unseen initial conditions. I now have a tool to visualize the motion of the foot as well as the ability to use the neural network with Chrono to predict the GRF. This weeks experiment shows that the two tools do as I expected them to do. 

Compare neural network's predicted GRF with Dan's RFT-based GRFs?

