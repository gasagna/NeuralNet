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
import cPickle as pickle
import numexpr as ne
import copy
import sys

def _myhstack(a,b):
    """Stack two arrays side by side"""
    c = np.empty( (a.shape[0], a.shape[1]+b.shape[1]) )
    c[:,:a.shape[1]] = a
    c[:,a.shape[1]:] = b
    return c

def sigmoid(x, beta=1):
    """Sigmoid activation function.
    
    The sigmoid activation function :math:`\\sigma(x)`  is defined as:
    
    .. math::
       \\sigma(x) = \\frac{1}{ 1+ \\exp( -\\beta x )}
       
    Parameters
    ----------
    
    x : float
    
    beta : float, default=1
      a parameter determining the steepness of the curve in :math:`x=0`
      
    Returns
    -------
    s : float
      the value of the sigmoid activation function
    """
    return ne.evaluate( "1.0 / ( 1 + exp(-beta*x))" )

def load_net_from_file( filename ):
        """Load net from a file."""
        return pickle.load(open(filename, 'r'))
        
        
class Dataset( object ):
    def __init__ ( self, inputs, targets ):
        
        #Check that the dataset is consistent
        if not inputs.ndim == 2:
            raise ValueError( 'inputs should be a two dimensions array.' )
            
        if not targets.ndim == 2:
            raise ValueError( 'targets should be a two dimensions array.' )
            
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError( 'the length of the inputs is not consistent with length of the targets.' )
        
        # set attributes
        self.inputs = inputs
        self.targets = targets
        
        # length of the dataset, number of samples
        self.n_samples = self.inputs.shape[0]
        
    def split( self, fractions=[0.5, 0.5]):
        """Split randomly the dataset into smaller dataset.
        
        Parameters
        ----------
        fraction: list of floats, default = [0.5, 0.5]
            the dataset is split into ``len(fraction)`` smaller
            dataset, and the ``i``-th dataset has a size
            which is ``fraction[i]`` of the original dataset.
            Note that ``sum(fraction)`` can also be smaller than one
            but not greater.
            
        Returns
        -------
        subsets: list of :py:class:`nn.Dataset`
            a list of the subsets of the original datasets
        """
        


        if sum(fractions) > 1.0 or sum(fractions) <= 0:
            raise ValueError( "the sum of fractions argument should be between 0 and 1" )
        
        # random indices
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)
        
        # insert zero
        fractions.insert(0, 0)
        
        # gte limits of the subsets
        limits = (np.cumsum(fractions)*self.n_samples ).astype(np.int32)
                
        subsets = []
        # create output dataset
        for i in range(len(fractions)-1):
            subsets.append( Dataset(self.inputs[idx[limits[i]:limits[i+1]]], self.targets[idx[limits[i]:limits[i+1]]]) )
        
        return subsets

    def __len__(self):
        return len( self.inputs )

class MultiLayerPerceptron( ):
    """A Multi Layer Perceptron feed-forward neural network.
    
    This class implements a multi layer feed-forward neural 
    network with hidden sigmoidal units and linear output 
    units. Multiple hidden layers can be defined and each 
    layer has a bias node. 
    """

    def __init__ ( self, arch, b=0.1, beta=1, n_threads=1 ):
        """Create a neural network.
        
        Parameters
        ----------
        arch : list of integers
            a list containing the number of neurons in each layer
            including the input and output layers.
            
        b : float, default = 0.1
            left/right limits of the uniform distribution from which 
            the intial values of the network weights are sampled.

        beta : float, default = 1
            steepness of the sigmoidal function in ``x=0``
            
        n_threads : int
            the number of threads to use when computing some expression
            with numexpr, if it is available on the system.
            
        Attributes
        ----------
        arch : list of integers
            a list containing the number of neurons in each layer
            including the input and output layers.
        
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
        # set attributes
        self.arch = arch
        self.beta = beta
        self.n_layers = len(arch)
        self.n_hidden = len(arch) - 2
        
        # set number of threads
        ne.set_num_threads(n_threads)
        
        # a list of arrays containing the weight of each layer
        # e.g. if arch = [ 2, 5, 1 ] then 
        # self.weights = [ shape(2+1, 5), shape(5+1, 1) ]
        self.weights = [ ]

        # init the weigths to small values
        for i in xrange( self.n_layers - 1 ):
            size = arch[i] + 1 , arch[i+1]
            self.weights.append( np.random.uniform(-b, b, size) )

    def save( self, filename ):
        """Save net to a file. The standard library cPickle module
        is used to store the network.
        
        Parameters
        ----------
        filename : str
            the file name to which to save the network.
        """
        clone = copy.copy( self )
        del clone._hidden
        pickle.dump( clone, open(filename, 'w') )

    def _forward_train ( self, dataset ):
        """Compute network output. This method is used only for training
        the network and should prefereably not be used after training. 
        This methods does bookkeeping of the activation values of the nodes,
        so that backprogation can work.
        

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
        self._check_dataset( dataset )
        
        # add biases values 
        hidden = _myhstack( dataset.inputs, -np.ones((dataset.n_samples,1)) )

        # keep track of the forward operations
        self._hidden = [ hidden ]
       
        # for each layer except the output, compute activation
        #  adding the biases as necessary
        for i in xrange( self.n_layers - 2 ):
            hidden = np.dot( hidden, self.weights[i] )
            hidden = sigmoid( hidden, self.beta )
            hidden = _myhstack( hidden, -np.ones( (hidden.shape[0],1) ) )
            
            self._hidden.append( hidden )
            
        # compute output
        return np.dot( hidden, self.weights[-1] )
        
    def forward ( self, dataset ):
        """Compute network output.
        
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
        self._check_dataset( dataset )
        
        # add biases values 
        hidden = _myhstack( dataset.inputs, -np.ones((dataset.n_samples,1)) )
      
        # for each layer except the output, compute activation
        #  adding the biases as necessary
        for i in xrange( self.n_layers - 2 ):
            hidden = np.dot( hidden, self.weights[i] )
            hidden = sigmoid( hidden, self.beta )
            hidden = _myhstack( hidden, -np.ones( (hidden.shape[0],1) ) )
            
        # compute output
        return np.dot( hidden, self.weights[-1] )

    def train_backprop ( self, training_set, validation_set=None, eta=0.5, alpha=0.5, n_iterations=100, etol=1e-6, verbose=True, k=0.01 ):
        """Train the network using the back-propagation algorithm.
        
        Training is performed in batch mode, i.e. all input samples are presented 
        to the network before an update of the weights is computed.
        
        Training by back-propagation is slow, since it is a first order method; 
        don't expect too much.
        
        
        Parameters
        ----------
        training_set : instance of :py:class:`nn.Dataset` class
            the keep the inputs and the targets used to train the network
                
        validation_set : instance of :py:class:`nn.Dataset` class
            the set of inputs/targets used to  validate the network training.
            If it is None, no validation is performed.
            
        eta : float, default=0.5
            the initial learning rate used in the training of the network.
            The learning rate decreases over time, when fluctuations
            occur in the mean square error  of the network.        
            
        alpha : float, default=0.5
            the momentum constant of the momentum term in the backpropagation
            algorithm. Must be between 0 and 1. If set to 0 no momentum
            is used.

        n_iterations : int, default=100
            the number of epochs of the training. All the input samples
            are presented to the network this number of times.

        etol : float, default = 1e-6
            training is stopped if difference between the error at successive 
            epochs is less than this value.
        
        verbose : bool, default is True
            whether to print some debugging information at each epoch.
            
        k : float
            a number defining the rate of change of the learning parameter ``eta``.
            If the error is decreasing the learning parameter is increased 
            as``eta *= 1 + k/10``, if it is increasing the learning parameter 
            is decreased as ``eta /= 1 + k``.
        
        """
        # check shape of the data
        self._check_dataset( training_set )
        if validation_set:
            self._check_dataset( validation_set )

        # initialize deltas 
        deltas = [ None ] * ( self.n_layers - 1 )
        
        # save errors at each iteration to plot convergence history
        err_save = np.zeros( n_iterations+1 )
        err_save[0] = self.error( training_set )
        
        # save also error over the validation set, if needed
        if validation_set:
            err_val_save = np.zeros( n_iterations+1 )
            err_val_save[0] = self.error( validation_set )

        # initialize weights at previous step
        d_weights_old = [ np.zeros_like(w) for w in self.weights ]
        
        # repeat the training
        for n in xrange( 1, n_iterations+1 ):

            # compute output                
            o = self._forward_train( training_set )

            # compute output delta term
            deltas[-1] = ( o - training_set.targets )

            # calculate deltas, for each hidden unit
            for i in xrange( self.n_hidden ): 
            # j is an index which go backwards
                j = -( i+1 )
                deltas[ j-1 ] = self._hidden[j][:,:-1] * ( 1.0 - self._hidden[j][:,:-1] ) *  np.dot( deltas[j], self.weights[j][:-1].T )
        
            # update weights
            for i in xrange( self.n_layers - 1 ):
                d_weights_old[i] = alpha*d_weights_old[i] + eta * np.dot( deltas[i].T, self._hidden[i] ).T / training_set.n_samples
                self.weights[i] -= d_weights_old[i]

            # save error
            err_save[n] = self.error( training_set )
            if validation_set:
               err_val_save[n] = self.error( validation_set )
                
            # break if we are close to the minimum
            if np.abs(err_save[n] - err_save[n-1]) < etol:
                print "Minimum variation of error reached. Stopping training."
                break
                
            # stop training if validation error is growing too much
            # look at the last 5 errors. if they are all increaing stop training.
            if validation_set:
                if np.all( np.diff(err_val_save[n-5:n]) > 0):
                    break
            
            # check error behaviour and change learning parameter
            if err_save[n] > err_save[n-1]:
                eta /= 1 + k 
            else:
                eta *= 1 + float(k) / 10
           
            # compute if error is growing or decreasing
            if err_save[n] >  err_save[n-1]:
                err_tr_sign = '(+)'
            elif err_save[n] <= err_save[n-1]:
                err_tr_sign = '(-)'
            if validation_set:
                if err_val_save[n]  > err_val_save[n-1]:
                    err_val_sign = '(+)'
                elif err_val_save[n] <= err_val_save[n-1]:
                    err_val_sign = '(-)'
            
            # print state information
            if validation_set:
                sys.stdout.write( '\b'*68 )
                sys.stdout.write( "Epoch %5d; MSE-tr = %6.6e%s; MSE-val = %6.6e%s" % (n, err_save[n], err_tr_sign, err_val_save[n], err_val_sign) )
            else:
                sys.stdout.write( '\b'*68 )
                sys.stdout.write( "Epoch %5d; MSE-tr = %6.6e%s" % (n, err_save[n], err_tr_sign) )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
               
        return err_save

    def train_quickprop ( self, training_set, validation_set=None, n_iterations=100, mu=1.5, etol=1e-6, epochs_between_reports=1 ):
        """Train the network using the quickprop algorithm.
        
        Training is performed in batch mode, i.e. all input samples are presented 
        to the network before an update of the weights is computed.
                
        Parameters
        ----------
        training_set : instance of :py:class:`nn.Dataset` class
            the set of inputs/targets used to train the network
                
        validation_set : instance of :py:class:`nn.Dataset` class
            the set of inputs/targets used to  validate the network training.
            If it is None, no validation is performed.

        n_iterations : int, default=100
            the number of epochs of the training. All the input samples
            are presented to the network this number of times.

        etol : float, default = 1e-6
            training is stopped if difference between the error at successive 
            epochs is less than this value.
        
        epoch_between_reports : int, default = 1
            report error every # of epochs
        
        References
        ----------
        [1] Neural Processing Letters 12: 159-169, 2000.
        Globally Convergent Modification of the Quickprop Method
        MICHAEL N. VRAHATIS, GEORGE D. MAGOULAS and VASSILIS P. PLAGIANAKOS
       
        """
        
        # check shape of the data
        self._check_dataset( training_set )
        if validation_set:
            self._check_dataset( validation_set )
        
        # initialize deltas list        
        deltas = [ None ] * ( self.n_layers - 1 )
        
        # save errors at each iteration to plot convergence history
        err_save = np.zeros( n_iterations+1 )
        err_save[0] = self.error( training_set )
        
        # save also error over the validation set, if needed
        if validation_set:
            err_val_save = np.zeros( n_iterations+1 )
            err_val_save[0] = self.error( validation_set )

        # initialize weights at previous step
        d_weights_old = [ np.zeros_like(w) for w in self.weights ]
        S_old = [ np.zeros_like(w) for w in self.weights ]
        
        # repeat the training for a certain amount of epochs
        for n in xrange( 1, n_iterations+1 ):

            # compute output                
            o = self._forward_train( training_set )

            # compute output delta term
            deltas[-1] = ( o - training_set.targets )

            # calculate deltas, for each hidden unit
            for i in xrange( self.n_hidden ): 
                # j is an index which goes backwards
                j = -( i+1 )
                a = self._hidden[j][:,:-1]
                b = np.dot( deltas[j], self.weights[j][:-1].T )
                deltas[ j-1 ] = a*(1-a)*b

            for i in xrange( self.n_layers - 1 ):
                # compute error derivative 
                S = np.dot( deltas[i].T, self._hidden[i] ).T / training_set.n_samples

                if n < 2:
                    # perform conventional backpropagation at first interation to ignite quick propagation algorithm
                    d_weights = 0.01 * S
    
                else:
                    # use quickprop algorithm 
                    d_weights =  mu * S / np.abs(S_old[i] - S) * np.abs(d_weights_old[i])
    
                    # check that we do not make a too large step
                    d_weights = np.where( np.abs(d_weights) > np.abs(mu*d_weights_old[i]), mu*d_weights_old[i], d_weights )

                # update weights
                self.weights[i] -= d_weights

                # keep track of stuff at previous step
                S_old[i] = S
                d_weights_old[i] = d_weights


            # debug message
            if n % epochs_between_reports == 0 and n >= epochs_between_reports :
                # compute error
                err_save[n] = self.error( training_set )
                if validation_set:
                    err_val_save[n] = self.error( validation_set )
                
                # compute if error is growing or decreasing
                if err_save[n] >  err_save[n-1]:
                    err_tr_sign = '(+)'
                elif err_save[n] <= err_save[n-1]:
                    err_tr_sign = '(-)'
                if validation_set:
                    if err_val_save[n]  > err_val_save[n-1]:
                        err_val_sign = '(+)'
                    elif err_val_save[n] <= err_val_save[n-1]:
                        err_val_sign = '(-)'
                
                # print state information
                if validation_set:
                    sys.stdout.write( '\b'*68 )
                    sys.stdout.write( "Epoch %5d; MSE-tr = %6.6e%s; MSE-val = %6.6e%s" % (n, err_save[n], err_tr_sign, err_val_save[n], err_val_sign) )
                else:
                    sys.stdout.write( '\b'*68 )
                    sys.stdout.write( "Epoch %5d; MSE-tr = %6.6e%s" % (n, err_save[n], err_tr_sign) )
                sys.stdout.flush()
                
                # break if we are close to the minimum
                if np.abs(err_save[n] - err_save[n-1]) < etol:
                    print "Minimum variation of error reached. Stopping training."
                    break
                
                # stop training if validation error is growing too much
                # look at the last 5 errors. if they are all increaing stop training.
                if validation_set:
                    if np.all( np.diff(err_val_save[n-5:n]) > 0):
                        break
                    
            else:
                err_save[n] = err_save[n-1]
                if validation_set:
                    err_val_save[n] = err_val_save[n-1]
            
        sys.stdout.write('\n')
        sys.stdout.flush()
                
        if validation_set:
            return err_save, err_val_save
        else:
            return err_save

    def _check_dataset( self, dataset ):
        """Check that the dataset is consistent with respect 
        to the  network architecture."""
        if not isinstance( dataset, Dataset ):
            raise ValueError( 'wrong training_set or validation_set are not instances of the nn.Dataset class' )
            
        if dataset.inputs.shape[1] != self.arch[0]:
            raise ValueError( 'dataset inputs shape is inconsistent with number of network input nodes.' )
        
        if dataset.targets.shape[1] != self.arch[-1]:
            raise ValueError( 'dataset targets shape is inconsistent with number of network output nodes.' )

    def error( self, dataset ):
        """Compute the sum of squared error of the network.
        
        The error of a network with :math:`n` output units, when
        given :math:`N` training samples, is defined as:
        
        .. math ::
        E = \\frac{1}{N} \\sum_i=1^N \sum_j=1^n ( o_j - t_j )^2
        
        Parameters
        ----------
        dataset : instance of :py:class:`nn.Dataset`
        
        Returns
        -------
        e : float 
            the mean square error of the network
        """
        return np.mean( (self.forward( dataset ) - dataset.targets)**2 )
