# Description
This week I tried to create a dataset with many trials, each with a diffrent (attack angle, intrusion angle) pair. 

Things I did this week on route to creating these graphs. I modified the Chrono simulation to allow me to set the angle of attack and the intrusion angle. I then added command line arguments so that I can run an experiment with variable attack angle and instrusion angle. I then wrote a bash script that runs the Chrono experiment many times with variable attack angles and intrusion angles. I also re-read the Chen Li paper and Zoomed Dan with the few questions I had about it.

# Setbacks
I opened up the file where I was storing the 3 vectors at each time step of the GRF. They are all 0. I then checked the demo which Juntao gave me to build on top of and that demo also has all (0, 0, 0) GRF. I looked into the API but could not sort it out. There must be some or a few small things I am doing wrong. I will work on resolving this. I spent a lot of time trying to understand the API and why I was getting (0, 0, 0) GRF     
