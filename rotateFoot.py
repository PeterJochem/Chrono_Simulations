import numpy as np
from graphics import *
import random
import time

# The window object we will write to
window_width = 800
window_height = 800
win = GraphWin("Visualization", 800, 800)
win.setBackground("white")

# This class describes the foot
class Foot:
    
    # Constructor
    def __init__(self):
        
        # These are the height and width of the foot
        self.length = 100.0
        self.height = 20.0

        self.center = np.zeros((3, 1))
        self.center[2][0] = 1.0

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
        self.point3[0][0] = -1 * self.length / 2.0
        self.point3[1][0] = -1 * self.height / 2.0
        self.point3[2][0] = 1.0 
        
        
        self.point4 = np.zeros((3, 1))
        self.point4[0][0] = self.length / 2.0
        self.point4[1][0] = -1 * self.height / 2.0
        self.point4[2][0] = 1.0

        # Store refrences to the graphics objects
        self.point1G = None
        self.point2G = None
        self.point3G = None
        self.point4G = None
        
        # Store refrences to the grpahics objects that 
        # are the lines between the points
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.line4 = None


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
        
        center = np.matmul(R, self.center)

        return point1, point2, point3, point4, center

    
    # Describe this method here
    def drawFoot(self, angle, x, y):
        
        # Describe 
        pt1, pt2, pt3, pt4, center = self.rotatePoints(angle, x, y) 

        # Turn the points into the plane frame
        self.point1G = Point(pt1[0][0], pt1[1][0] )
        self.point2G = Point(pt2[0][0], pt2[1][0] )
        self.point3G = Point(pt3[0][0], pt3[1][0] )
        self.point4G = Point(pt4[0][0], pt4[1][0] )

        center = Point(center[0][0], center[1][0])
        center = Circle(center, 3.5)
        center.setWidth(1)
        center.setFill("red")
        center.draw(win)

        # Draw each point to the screen
        self.point1G.draw(win)
        self.point2G.draw(win)
        self.point3G.draw(win)
        self.point4G.draw(win)
   
        # Draw line between the points
        self.line1 = Line(self.point1G, self.point2G)
        self.line2 = Line(self.point2G, self.point3G)
        self.line3 = Line(self.point3G, self.point4G)
        self.line4 = Line(self.point4G, self.point1G)
        
        # Set the line's thickness
        self.line1.setWidth(5)
        self.line2.setWidth(5)
        self.line3.setWidth(5)
        self.line4.setWidth(5)
        
        # Set the line's color
        self.line1.setFill("blue")
        self.line2.setFill("blue")
        self.line3.setFill("blue")
        self.line4.setFill("blue")

        self.line1.draw(win)
        self.line2.draw(win)
        self.line3.draw(win)
        self.line4.draw(win)
    

    # Describe this method here
    def undrawFoot(self):
        
        # Undraw the lines
        self.line1.undraw()
        self.line2.undraw()
        self.line3.undraw()
        self.line4.undraw()
    
        # Undraw the points
        self.point1G.undraw()
        self.point2G.undraw()
        self.point3G.undraw()
        self.point4G.undraw()
   

# Describe this metod here
# Inputs: 
# Returns: 
def getNextState(fp, skip):
    
    # Skip over certain number of states
    for i in range(skip):
        skipOver = fp.readline()

    myLine = fp.readline()
    
    # Parse the data out of the string
    myLine = myLine.split(",") 
    
    angle = float( myLine[1] )  
    
    # Center the position to the center 
    x = float( myLine[2] ) + (window_width / 2.0)

    # Center the position to the center
    y = float( myLine[6] ) + (window_height / 2.0)

    print(str("( ") + str(x) + ", " + str(y) + str(")") )

    return angle, x, y
    
# Read next foot state
# Animate State

# The file of points describing the foot's state
fp = open('sim_data/output_plate_positions_and_velocities.csv', 'r')

i = 0
myFoot = Foot()
while(True):

    # Get next foot state
    # angle, x, y = getNextState(fp, 2000)

    # Draw objects 
    myFoot.drawFoot( (2 * np.pi) * (i /360.0), i, (window_height / 2.0) + (np.sin(i/5.0) * window_height / 20.0 ) )

    # Pause so person can visualize
    time.sleep(0.025)
    myFoot.undrawFoot()

    i = i + 1


fp.close()



