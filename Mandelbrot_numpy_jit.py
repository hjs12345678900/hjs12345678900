import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as tm
from numba import jit

'''
In this program, numba and np.array is utilised to do calculation for the image (as a matrix). This method is
working pretty well from my point of view, and it can produce width = 1000, iterations of 50 in 4 sec (in the jupyter
notebook this number is 10-12 for my configuration)

The use of matrix manipulation allows higher CPU usage in higher width, but it comes with a higher usage for RAM (as
the matrix gets very large) My laptop can handle up till 15000 ish width
'''


ss = tm.time()

# For the whole graph
xrange = np.array([-2,2])
yrange = np.array([-2,2])

width = 2000
N_iterations = 1000

# Setting up the complex numbers
real_col = np.array([np.linspace(xrange[0],xrange[1],width)])
im_col = np.array([np.linspace(yrange[0],yrange[1],width)])

real = np.ones((width,1)).dot(real_col)
im = np.transpose(im_col).dot(np.ones((1,width)))

print('real and im prepared')

@jit(nopython = True)
def vals_i_prepare(real,im):
    vals_i = real + 1j *im

    return vals_i

vals = vals = np.zeros((width,width))
vals_i = vals_i_prepare(real,im)

print('vals prepared')


# Start of the caluclation section
@jit(nopython = True)
def manipulation(vals,vals_i):
    vals= vals*vals + vals_i
    return vals

vals_convergence = np.zeros((width,width))

for n in range(N_iterations):
    start = tm.time()
    vals_old = np.abs(vals) < 2
    vals = manipulation(vals,vals_i)

    vals_new = np.abs(vals) > 2
    vals_change = vals_old & vals_new
    vals_nan_change = np.isnan(vals_new) & (np.isfinite(vals_old))
    vals_convergence[(vals_change)|(vals_nan_change)] = (1- n/(N_iterations))
    end = tm.time()
    #if n in np.arange(0,10001,1):
        #print(n,end-start)

ee = tm.time()
print('total time is', ee-ss)


# Plotting
cscheme = 'gist_ncar'
crange = (0,1)
ax = plt.gca()
ax.axis('equal')
ax.set_aspect('equal','box')
pcol = plt.imshow(np.flipud(vals_convergence),cmap = cscheme,vmin = crange[0],vmax = crange[1], extent=[xrange[0],xrange[1],yrange[0],yrange[1]])
cmap = matplotlib.cm.ScalarMappable()
cmap.set_clim(vmin = -crange[0],vmax = crange[1])
cbar = plt.colorbar()
plt.show()



