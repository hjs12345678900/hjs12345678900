import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as tm

ss = tm.time()

# This method should convert a pixel index to a position for a plotting range given by bounds
def index_to_position(index,width,bounds = [-2,2]):
    return bounds[0]+index*((bounds[1] - bounds[0])/(width-1))

# N_iterations is the number of times the iteration is performed at each pixel. If |z|<2 after N_iterations, the 
# set is thought to be in the set
N_iterations = 100

# This specifies the number of pixels that you'll divide your space into
width = 1001

# In the first case vals will be an array of ones and zeros to allow you to plot points which are inside the set
# and points which are outside of the set. Once that is working, vals will vary between 0 and 1 to give an indication
# of how rapidly |z| diverges when it is not in the set.
vals = np.zeros((width,width))
xrange = np.array([-2,2])
yrange = np.array([-2,2])


for i in range(width):
    x = index_to_position(i,width,bounds = xrange)
    for j in range(width):
        y = index_to_position(j,width,bounds = yrange)
        c = complex(x,y)
        z = complex(0,0)
        for k in range(N_iterations):
            # This is where each position is checked to find if it is in the Mandelbrot set.
            z = z*z + c

            if abs(z) > 2:
                vals[j,i] = 1 - k/N_iterations
                break   # Breaak the for loop to assign the correct value and 
                        # avoid overflow


ee = tm.time()
print('total time is:',ee-ss)

# plotting the set.
cscheme = 'hot'
crange = (0.,0.6)
ax = plt.gca()
ax.axis('equal')
ax.set_aspect('equal','box')
pcol = plt.imshow(np.flipud(vals),cmap = cscheme,vmin = crange[0],vmax = crange[1], extent=[-2,2,-2,2])
cmap = matplotlib.cm.ScalarMappable()
cmap.set_clim(vmin = -crange[0],vmax = crange[1])
cbar = plt.colorbar()


plt.show()