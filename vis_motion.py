import numpy as np
from graphics import *
import random


class Foot:

    def __init__(self):
    
        # These are the four points of the foot
        self.point1 = np.zeros((1, 2)) 
        self.point2 = np.zeros((1, 2))
        self.point3 = np.zeros((1, 2))
        self.point4 = np.zeros((1, 2))

# Read next foot state
# Animate State
win = GraphWin()

pt = Point(100, 50)

pt.draw(win)

while(True):
    # Get next foot state

    # Create graphics objects needed 

    # Draw objects 

    # Pause so person can visualize
