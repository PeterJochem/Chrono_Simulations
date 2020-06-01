# Summary 
I wanted to generate a similiar dataset to the one Juntao generated to make graphs 2A-D in Chen Li's paper. This required me to read and understand the Chrono simulation. I ran into a few errors which I could not resolve. The longer description elaborates. I talked to Dan and Juntao and it likely makes more sense if instead of re-generating data similiar to Juntao's, I use his data. The Chrono code is tough to modify and if Juntao understands it quite well, I am proabbly more useful spending time buildig out the machine learning API. I have part of Juntao's data which varies (attack angle, intrusion angle) and was used to replicate Chen Li's graphs. From this, I trained the network to map (x, y, z, attackAngle, intrusionAngle) to (F_x, F_y, F_z). From this, I can compute the desired N/cm^3 

# Plan For Next Week
Use Juntao's data to learn the same function and compare it to Chen Li's and Juntao's graphs
Dan asked for a description of the database along with a pdf of how to use it

# More Detailed Summary
This week I tried to create a dataset with many trials, each with a diffrent (attack angle, intrusion angle) pair. 
Things I did this week on route to creating these graphs. I modified the Chrono simulation to allow me to set the angle of attack and the intrusion angle. I then added command line arguments so that I can run an experiment with variable attack angle and instrusion angle. I then wrote a bash script that runs the Chrono experiment many times with variable attack angles and intrusion angles.

# Setbacks
I opened up the file where I was storing the 3 vectors at each time step of the GRF. They are all 0. I then checked the demo which Juntao gave me to build on top of and that demo also has all (0, 0, 0) GRF. I looked into the API but could not sort it out. There must be some or a few small things I am doing wrong. I will work on resolving this. I spent a lot of time trying to understand the API and why I was getting (0, 0, 0) GRF. I retraced my steps by copying the most recent simulation code and modifying it to suit my needs.

