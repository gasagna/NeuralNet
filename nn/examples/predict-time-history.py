import os
import numpy as np
from nn import MultiLayerPerceptron
from pyhd.sig import Signal
from pylab import *

#from pyhd.hw import H5Data, SplineCalibration, read_data
## load calibration data
#datadir = '/home/davide/Documents/dottorato/data-lab/hw-monte-valle/data/calibrations/tar0224-0232'

#try:
    #e, v = np.loadtxt( os.path.join(datadir, 'ev.txt'), unpack=True )
#except IOError:
    #read_data( datadir, p_amb=99000, T_amb=24.4)
        

## create calibration object
#cal = SplineCalibration( e, v )

## get data
#full_data = H5Data( '/home/davide/Documents/dottorato/data-lab/hw-monte-valle/data/slprofile057.h5', cal, 0 )

# get velocity time history
#u = full_data.get_velocity_time_history( 2 )


u = np.load('u-y=2-slprofile061.npy')

s = Signal(u,11000)
s.detrend()
s.apply_obukhov_decomposition(2000, 'low')

u = s.data

# create time array for plotting
t = np.arange(len(u)) / 11000.0


#########################
# Neural network stuff

# number of input nodes
Ni = 150

# number of hidden nodes
Nh = 30

# number of output nodes
No = 100

# archtecture of the net
if Nh == 0:
    arch = [Ni, No]
else:
    arch = [Ni, Nh, 30, No]


# pack data to train the network
u_pack = u[:(Ni+No)*(len(u)/(Ni+No))].reshape( -1, Ni+No )

# 
frac = 0.1
n_samples = int( frac * u_pack.shape[0] )
#n_samples = 

# choose randomly the input dataset among the complete dataset 
# since we are sampling integers it is possible that we sample
# two time the same number and so we eliminate it. It that case
# the number of samples could be slightly smaller than what
# previously set.
#idx = np.unique( np.random.randint(0, u_pack.shape[0], n_samples))
idx = np.arange( n_samples  )

# inputs are all (Ni) columns except the last, which will be the targets (No)
inputs  = u_pack[idx, :Ni]
targets = u_pack[idx, Ni:].reshape(-1,No)


# create neural network
net = MultiLayerPerceptron( arch, eta=0.3, sigma=1)

try:
    # train it
    net.train( inputs, targets, n_iterations=1000000, etol=1e-18 )
except KeyboardInterrupt:
    pass

print "Number of samples = %d " % n_samples
print "Net architecture %s " % arch


n = 3
fig = figure( figsize=(15,8))
plt.subplots_adjust(left=0.05, right=0.96, hspace=0.2, wspace=0.15)
for i in range( n**2 ):
    ax = fig.add_subplot( n, n, i+1 )
        
    plot( targets[i], 'k-', lw=3 )
    plot( net.forward( inputs[i,:].reshape(1,-1) )[0], 'r--', lw=1.2 )
    
    if i in [6, 7, 8]:
        xlabel('k')
    if i in [0, 3, 6]:
        ylabel('u [m/s]')
    #xlim(0,1)
    ylim(-0.4, 0.4)
    #ax.axis('tight')
    grid(1)

savefig('/home/davide/Documents/dottorato/corsi/Apprendimento-mimetico/nn/short-arch=[%d-%d-%d].png' % (arch[0], arch[1], arch[2]), dpi=250)
#show()


