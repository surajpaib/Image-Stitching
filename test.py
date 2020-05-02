import numpy

world = numpy.array([
    [0,0,0],
    [7,0,0],
    [0,5,0],
    [7,4,0]])
    
camera = numpy.array([
    [582,344],
    [834,338],
    [586,529],
    [841,522]])
    
#Lose Z axis
world = world[:,0:2]

#Make a square matrix
A = numpy.vstack([world.T, numpy.ones(4)]).T

#perform the least squares method

print(A.shape, camera.shape)
x, res, rank, s = numpy.linalg.lstsq(A, camera, rcond=None)

#test results
print(x)