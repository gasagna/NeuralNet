import os

import numpy as np
from pylab import *

from nn import MultiLayerPerceptron


# load data from file
u = np.loadtxt('sample-data.txt')

# remove mean value from data
u -= u.mean()


#########################
# Neural network stuff
# number of input nodes
Ni = 50

# number of hidden nodes
Nh = 10

# number of output nodes
No = 50

# architecture of the net
arch = [Ni, Nh, No]


####################################
# Pack data into inputs and targets
# pack data to train the network
u_pack = u[:(Ni+No)*(len(u)/(Ni+No))].reshape( -1, Ni+No )

# only train over a fraction of the complete data set
frac = 0.1
n_samples = int( frac * u_pack.shape[0] )

# so use the first n_samples to train the net
idx = np.arange( n_samples  )

# unpack the data into input and outputs
inputs  = u_pack[idx, :Ni]
targets = u_pack[idx, Ni:].reshape(-1,No)



# create neural network
net = MultiLayerPerceptron( arch, eta=0.6, sigma=1 )

# train it. If Ctrl+c is pressed during the training stop it
try:
    net.train( inputs, targets, n_iterations=1000000, etol=1e-18 )
except KeyboardInterrupt:
    pass

# plot some results
n = 2
fig = figure( figsize=(15,8))
plt.subplots_adjust(left=0.05, right=0.96, hspace=0.2, wspace=0.15)
for i in range( n**2 ):
    ax = fig.add_subplot( n, n, i+1 )
        
    plot( targets[i], 'k->', lw=3, label='Target' )
    plot( net.forward( inputs[i,:].reshape(1,-1) )[0], 'ro', lw=1.2, label='Net prediction' )
    
    if i in [2,3]:
        xlabel('time step')
    if i in [0, 1]:
        ylabel('u [m/s]')
    ylim(-0.4, 0.4)
    grid(1)
    legend()

show()

