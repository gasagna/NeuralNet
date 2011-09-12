__author__ = "Davide Lasagna, Dipartimento di Ingegneria Aerospaziale, Politecnico di Torino"

__licence_ = """
Copyright (C) 2011  Davide Lasagna

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

def sigmoid(x, beta=1):
    """Sigmoid activation function.
    
    The sigmoid activation function :math:`\\sigma(x)`  is defined as:
    
    .. math::
       \\sigma(x) = \\frac{1}{ 1+ \\exp( -\\beta x )}
       
    Parameters
    ----------
    
    x : float
    
    beta : float, default=1
      a parameters determining the steepness of the curve in :math:`x=0`
      
    Returns
    -------
    s : float
      the value of the sigmoid activation function
    """
    return 1.0 / ( 1 + np.exp(-beta*x))

class MultiLayerPerceptron( ):
    """A Multi Layer Perceptron feed-forward neural network.
    
    This class implements a multi layer feed-forward neural 
    network with hidden sigmoidal units and linear output 
    units. Multiple hidden layers can be defined and each 
    layer has a bias node. The network is trained using 
    the Back-Propagation algorithm.
    """
    
    def __init__ ( self, arch, eta=0.5, sigma=0.1 ):
        """Create a neural network.
        
        Parameters
        ----------
        arch : list of integers
            a list containing the number of neurons in each layer
            including the input and output layers.
            
        eta : float, default=0.5
            the initial learning rate used in the training of the network
            
        sigma : float, default = 0.1
            standard deviation of the gaussian distribution from which 
            the intial values of the network weights are sampled.
            
        Attributes
        ----------
        arch : list of integers
            a list containing the number of neurons in each layer
            including the input and output layers.
        
        eta : float
            the initial learning rate used in the training of the network.
            The learning rate decreases over time, when fluctuations
            occur in the mean square error  of the network.
        
        n_layers : int
            the number of layers, including input and output layers
        
        n_hidden : int
            the number of hidden layers
        
        weights : list of np.ndarray
            a list of weights. Each weight array has a shape so that 
            the feed-forward step can be performed by matrix 
            multiplication: e.g. if ``net.arch`` is ``[5, 10, 10, 1]``
            then ``net.weights[0].shape`` is ``(10, 6)``, etc, ...
            
            
        Examples
        --------
        To define a network with 2 input units, 1 hidden layer
        with 4 units and a single output unit give:
        
        
        >>> import nn
        >>> net = nn.MultiLayerPerceptron( arch=[2, 4, 1] )
        
        Weights are initially set to small random values:
        
        >>> net.weights
        >>> [array([[ 0.78450808, -0.67768375,  0.54017801, -0.82977849,  0.03744147],                                                      
            [-0.25655105,  1.26595283, -1.23658886, -0.02595137, -0.5636417 ],                                                       
            [ 2.24445352, -0.6239078 , -1.51898104, -0.38226579,  0.78420179]]),
            array([[ 0.04237318],
            [-0.12053617],
            [ 1.81033076],
            [ 2.45879228],
            [ 0.3647032 ],
            [ 0.93829515]])]
        
        Define some input data:
        
        >>> y = np.array( [0,1] )
        
        Compute network output:
        
        >>> net.forward( y )
        
        """
        # the architecture of the network. 
        self.arch = arch
        self.eta = eta
        self.n_layers = len(arch)
        self.n_hidden = len(arch) - 2
        
        # a list of arrays containing the weight of each layer
        # e.g. if arch = [ 2, 5, 1 ] then 
        # self.weights = [ shape(2+1, 5), shape(5+1, 1) ]
        self.weights = [ ]

        # init the weigths to small values
        for i in range( self.n_layers - 1 ):
            size = arch[i] + 1 , arch[i+1]
            self.weights.append( np.random.normal(0, sigma, size) )

    def forward ( self, inputs ):
        """Compute network output.
        
        After the training the network can be used on new data.
        
        Parameters
        ----------
        inputs  :   np.ndarray
        must be a two dimensions array with shape equal to 
        ``(n_samples, net.arch[0])``.
        
        Returns
        ------- 
        output : np.ndarray
        a two dimensions array with shape ``(n_samples, net.arch[-1])``
        """
        
        # check shape of the data
        self._check_inputs( inputs )
        
        # add biases values 
        hidden = np.c_[ inputs, -np.ones(inputs.shape[0]) ]

        # keep track of the forward operations
        self._hidden = [ hidden ]
       
        # for each layer except the output, compute activation
        #  adding the biases as necessary
        for i in range( self.n_layers - 2 ):
            hidden = np.dot( hidden, self.weights[i] )
            hidden = sigmoid( hidden )
            hidden = np.c_[ hidden, -np.ones( hidden.shape[0] ) ]
            
            self._hidden.append( hidden )
            
        # compute output
        return np.dot( hidden, self.weights[-1] )

    def train ( self, inputs, targets, n_iterations=100, etol=1e-6, verbose=True ):
        """Train the network using the back-propagation algorithm.
        
        Training is performed in batch mode, i.e. all input samples are presented 
        to the network before an update of the weights is computed.
        
        Training by back-propagation is slow, since it is a first order method; 
        don't expect too much.
        
        
        Parameters
        ----------
        inputs  :   np.ndarray
            the input data of the training set. Must be a two dimensions array 
            with shape equal to ``(n_samples, net.arch[0])``.
        
        targets  :   np.ndarray
            the target data of the training set. Must be a two dimensions array
            with shape equal to ``(n_samples, net.arch[0])``.        
        
        n_iterations : int, default=100
            the number of epochs of the training. All the input samples
            are presented to the network this number of times.
        
        verbose : bool, default is True
            whether to print some debugging information at each epoch.
        
        """
        
        # check shape of the data
        self._check_inputs( inputs )
        self._check_targets( targets )
        
        err = self.error( inputs, targets )
        deltas = [ None ] * ( self.n_layers - 1 )
        
        # save errors at each iteration to plot convergence history
        err_save = np.zeros( (n_iterations,) )
        
        # repeat the training
        for n in xrange( n_iterations ):

            # compute output                
            o = self.forward( inputs )

            # compute output delta term
            deltas[-1] = ( o - targets )

            # calculate deltas, for each hidden unit
            for i in range( self.n_hidden ): 
            # j is an index which go backwards
                j = -( i+1 )
                deltas[ j-1 ] = self._hidden[j][:,:-1] * ( 1.0 - self._hidden[j][:,:-1] ) *  np.dot( deltas[j], self.weights[j][:-1].T )
        
            # update weights
            for i in range( self.n_layers - 2 ):
                self.weights[i] -= self.eta * np.dot( deltas[i].T, self._hidden[i] ).T / inputs.shape[0]

            # update outputs weights
            a, b = deltas[-1].T, self._hidden[-1]
            self.weights[-1] -= self.eta * np.dot( a, b ).T / inputs.shape[0]

            # save error
            err_save[n] = err
            
            err_old = err            
            err = self.error( inputs, targets )
            
            if np.abs(err - err_old) < etol:
                if verbose:
                    print "Minimum variation of error reached. Stopping training."
                break
            
            if err > err_old:
                self.eta /= 1.001
            else:
                self.eta *= 1.00001
            
            if verbose:
                print "%5d %6.3e %8.5f" % (n, err, self.eta)
                
        return err_save
                
    def _check_inputs( self, inputs ):
        """Check that the shape (dimensionality) of the inputs
        is correct with respect to the   network architecture."""
        if inputs.shape[1] != self.arch[0]:
            raise ValueError( 'inputs shape is inconsistent with number of input nodes.' )
            
    def _check_targets( self, targets ):
        """Check that the shape (dimensionality) of the targets
        is correct with respect to the   network architecture."""
        if targets.shape[1] != self.arch[-1]:
            raise ValueError( 'targets shape is inconsistent with number of output nodes.' )

    def error( self, inputs, targets ):
        """Compute the sum of squared error of the network.
        
        The error of a network with :math:`n` output units, when
        given :math:`N` training samples, is defined as:
        
        .. math ::
        E = \\frac{1}{N} \\sum_i=1^N \sum_j=1^n ( o_j - t_j )^2
        
        Parameters
        ----------
        inputs  :   np.ndarray
            the input data of the training set. Must be a two dimensions array 
            with shape equal to ``(n_samples, net.arch[0])``.
        
        targets  :   np.ndarray
            the target data of the training set. Must be a two dimensions array
            with shape equal to ``(n_samples, net.arch[0])``.       
        
        Returns
        -------
        e : float 
            the error of the network
        """
        return np.mean( (self.forward( inputs ) - targets)**2 )
