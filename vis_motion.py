import numpy as np
from graphics import *
import random
import time

# The window object we will write to
win = GraphWin("Visualization", 800, 800)

# This class describes the foot
class Foot:
    
    # Constructor
    def __init__(self):
        
        # These are the height and width of the foot
        self.length = 100.0
        self.height = 20.0

        # These are the four points of the foot
        self.point1 = np.zeros((3, 1))
        self.point1[0][0] = self.length / 2.0   
        self.point1[1][0] = self.height / 2.0
        self.point1[2][0] = 1.0

        self.point2 = np.zeros((3, 1))
        self.point2[0][0] = -1 * self.length / 2.0
        self.point2[1][0] = self.height / 2.0
        self.point2[2][0] = 1.0

        
        self.point3 = np.zeros((3, 1))
        self.point3[0][0] = self.length / 2.0
        self.point3[1][0] = -1 * self.height / 2.0
        self.point3[2][0] = 1.0 
        
        
        self.point4 = np.zeros((3, 1))
        self.point4[0][0] = -1 * self.length / 2.0
        self.point4[1][0] = -1 * self.height / 2.0
        self.point4[2][0] = 1.0

        # Store refrences to the graphics objects
        self.point1G = None
        self.point2G = None
        self.point3G = None
        self.point4G = None


    # Describe here
    def createRotationMatrix(self, angle, x, y):
        
        R = np.zeros((3, 3))
        R[0][0] = np.cos(angle) 
        R[0][1] = np.sin(angle)
        R[0][2] = 0.0 

        R[1][0] = -1 * np.sin(angle)
        R[1][1] = np.cos(angle)
        R[1][2] = 0.0

        R[2][0] = x 
        R[2][1] = y 
        R[2][2] = 1.0
    
        R = R.transpose()
        
        return R

    # Describe here
    # Inputs: 
    # Outputs: 
    def rotatePoints(self, angle, x, y):
        
        # Generate the rotation matrix
        R = self.createRotationMatrix(angle, x, y)

        # Rotate the first point into the world frame       
        point1 = np.matmul(R, self.point1)
        point2 = np.matmul(R, self.point2)
        point3 = np.matmul(R, self.point3)
        point4 = np.matmul(R, self.point4)

        return point1, point2, point3, point4

    
    # Describe this method here
    def drawFoot(self, angle, x, y):
        
        # Describe 
        pt1, pt2, pt3, pt4 = self.rotatePoints(angle, x, y) 
        
        # Turn the points into the plane frame
        self.point1G = Point(pt1[0][0], pt1[1][0] )
        self.point2G = Point(pt2[0][0], pt2[1][0] )
        self.point3G = Point(pt3[0][0], pt3[1][0] )
        self.point4G = Point(pt4[0][0], pt4[1][0] )

        # Draw each point to the screen
        self.point1G.draw(win)
        self.point2G.draw(win)
        self.point3G.draw(win)
        self.point4G.draw(win)
    
    # Describe this method here
    def undrawFoot(self):
        
        self.point1G.undraw()
        self.point2G.undraw()
        self.point3G.undraw()
        self.point4G.undraw()

    
# Read next foot state
# Animate State

# The file of points describing the foot's state
fp = open('sim_data/output_plate_positions_and_velocities.csv', 'r')

i = 0
myFoot = Foot()
while(i < 360):
    # Get next foot state
    # line = fp.readline()    
    
    # Draw objects 
    # Pause so person can visualize
    
    myFoot.drawFoot( (2 * np.pi) * (i / 360.0), 200, 200 )
    time.sleep(0.1)     
    myFoot.undrawFoot()

    i = i + 1

fp.close()



