import nn
import numpy as np
from pylab import *

# create a dataset
x = np.linspace(-1,1,1000)
# we want to learn this function using a neural network
y = np.exp(x)*np.sin(5*x) + np.random.normal(0,0.4,1000)

# pack data into a dataset
d = nn.Dataset( x.reshape(-1,1), y.reshape(-1,1) )

# create a neural network object
net = nn.MultiLayerPerceptron( arch=[1,8,1], beta=1, b=1 )
# train it
err = net.train_quickprop( d, d, n_iterations=800, mu=1.9, etol=1e-12, epochs_between_reports=20 )

# plot the result
plot( x, y, 'r.')
plot( x, net.forward(d).ravel(), 'k', lw=2 )
show()
