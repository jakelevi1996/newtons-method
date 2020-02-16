"""
Module which contains different objective functions. Each objective functions is
implemented as a class, which inherits from an abstract ObjectiveFunction
interface class.

Each objective function class should have methods for the objective function
itself, as well as the first and second derivatives, a __call__ method which is
a wrapper for the objective function, and optionally also first and second
derivatives calculated using only a subset of the available variables, which are
to be used for block optimisation routines.
"""
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

class ObjectiveFunction():
    """
    ObjectiveFunction: interface/parent class for all objective functions
    """

    def f(self, x):
        """
        f: objective function, evaluated at input vector

        Inputs: x: input parameters, in a numpy array with shape (n,)

        Outputs: f: objective function evaluated at x, as a numpy.float64
        """
        raise NotImplementedError

    def dfdx(self, x, f=None):
        """
        dfdx: gradient of the objective function with respect to the parameters,
        evaluated at input vector

        Inputs
        -   x: input parameters, in a numpy array with shape (n,)
        -   f (optional): the objective function evaluated at x; if it is used
            by this method and has already been calculated, it can be passed as
            an argument to prevent repeat calculation, as a numpy.float64

        Outputs
        -   f: gradient of the objective function evaluated at x, in a numpy
            array with shape (n,)
        """
        raise NotImplementedError

    def d2fdx2(self, x, f=None):
        """
        d2fdx2: Hessian of the objective function with respect to the
        parameters, evaluated at input vector

        Inputs
        -   x: input parameters, in a numpy array with shape (n,)
        -   f (optional): the objective function evaluated at x; if it is used
            by this method and has already been calculated, it can be passed as
            an argument to prevent repeat calculation, as a numpy.float64

        Outputs
        -   f: Hessian of the objective function evaluated at x, in a numpy
            array with shape (n, n)
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        __call__: wrapper for self.f(x)
        """
        return self.f(x)

class Gaussian(ObjectiveFunction):
    """
    Negative Gaussian function with zero mean and diagonal variance
    """
    def __init__(self, scale=None, ndims=None):
        if scale is not None:   self.scale = np.array(scale)
        elif ndims is not None: self.scale = np.ones(ndims)
        else: raise ValueError("Must specify scale or ndims")

    def f(self, x):
        return 1.0 - np.exp(-np.dot(self.scale * x, x))

    def dfdx(self, x, f=None, inds=None):
        # if f is None: f = self.f(x)
        if inds is not None: raise NotImplementedError
        return 2.0 * self.scale * x * np.exp(-np.dot(self.scale * x, x))

    def d2fdx2(self, x, f=None, inds=None):
        # if f is None: f = self.f(x)
        if inds is not None: raise NotImplementedError
        scaled_x = self.scale * x
        return -2.0 * np.exp(-np.dot(self.scale * x, x)) * (
            2.0*np.outer(scaled_x, scaled_x) - self.scale*np.identity(x.size)
        )
        

class Cauchy(ObjectiveFunction):
    def __init__(self):
        raise NotImplementedError


if __name__ == "__main__":
    og = Gaussian(ndims=2)
    x = np.ones(2)
    print(
        x, x.dot(x), np.exp(-x.dot(x)), og(x), og.dfdx(x), og.d2fdx2(x),
        *np.linalg.eigh(og.d2fdx2(x)), sep="\n"
    )
    x = np.zeros(2)
    print(og.d2fdx2(x))

    n = 200
    x = np.ones(n)
    og = Gaussian(ndims=n)
    for _ in range(100):
        og.f(x)
    t0 = perf_counter()
    for _ in range(1000):
        # og.f(x)
        og.d2fdx2(x)
    t1 = perf_counter()
    print("Time taken = {:.5f}".format(t1 - t0))
