import nn
import numpy as np
from pylab import *

# create a dataset
x = np.linspace(-1,1,1000)
# we want to learn this function using a neural network
y = np.exp(x)*np.sin(5*x) + np.random.normal(0,0.1,1000)

# create a neural network object
net = nn.MultiLayerPerceptron( arch=[1,4,1] )
# train it
net.train(x.reshape(-1,1), y.reshape(-1,1), 5000, etol=1e-12 )

# plot the result
plot( x, y, '-')
plot( x, net.forward(x.reshape(-1,1)).ravel(), '.' )
show()
