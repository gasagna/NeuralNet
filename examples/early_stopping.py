import nn
import numpy as np
from pylab import *

# create a dataset
x = np.linspace(-1,1,100)
# we want to learn this function using a neural network
y = np.exp(x)*np.sin(3*x) + np.random.normal(0,0.1,len(x))

# pack data into a dataset
d = nn.Dataset( x.reshape(-1,1), y.reshape(-1,1) )

# split data into training set and validation
train_set, val_set = d.split( [0.5, 0.5] )

# create a neural network object
net = nn.MultiLayerPerceptron( arch=[1,20,1], beta=0.01, b=0.01 )

# train it
err = net.train_backprop( train_set, val_set, n_iterations=30000, eta=0.5, alpha=0.9, etol=1e-1200, max_ratio=0.4 )

# plot the result
plot( x, y, 'r.')
plot( x, net.forward(d).ravel(), 'k', lw=2 )
show()
