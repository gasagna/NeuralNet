.. _tut:

Tutorial
========
In this tutorial we are going to see the basic capabilties of the neural scikit.Several other examples can be found in examples page, where more advanced techniques and corner cases are used.

Create a dataset
================
We usually have a set of input/output data and we would like to learn a complex non-linear mapping from this data using a neural network. So, we first have to pack our data into a convenient form, a :py:class:`nn.Dataset` object.
