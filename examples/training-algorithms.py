import nn
import numpy as np
from pylab import *

# create a dataset
x = np.linspace(-1,1,1000)

# generate some random noise with specified variance
sigma = 0.1

# we want to learn this function using a neural network
y = np.exp(x)*np.sin(10*x) + np.random.normal(0,sigma,1000)

# create a dataset from these data. We use all the data for training
d = nn.Dataset( x.reshape(-1,1), y.reshape(-1,1) )

# create a neural network object
net_bp = nn.MultiLayerPerceptron( arch=[1,10,1], beta=1, b=1 )
# train it using standard backpropagation with momentum
err_bp = net_bp.train_backprop(training_set=d, validation_set=None, n_iterations=10000, eta=0.8, alpha=0.8, etol=1e-16 )

# train it using quickprop
net_qp = nn.MultiLayerPerceptron( arch=[1,10,1], beta=1, b=1 )
err_qp = net_qp.train_quickprop(training_set=d, validation_set=None, n_iterations=10000, mu=1.6, etol=1e-16 )

# plot the result
figure( figsize=(10,7))
title('Convergence properties of standard back propagation and quick propagation')
loglog( err_bp, label='Back Propagation')
loglog( err_qp, label='Quick Propagation')
axhline(sigma**2, label='simulated random noise variance', color='k')
legend()
xlabel('number of epochs')
ylabel('mean square error')
ylim(0.5 * sigma**2, 1e2)
grid()

figure()
title('Network prediction and actual data')
plot( x, y, 'b.')
plot( x, net_bp.forward(d).ravel(), label='Back Propagation', color='r', ls='-', lw=2)
plot( x, net_qp.forward(d).ravel(), label='Quick Propagation', color='g', ls='-', lw=2 )
grid()
legend()
xlabel('x')
ylabel('y(x)')
show()
