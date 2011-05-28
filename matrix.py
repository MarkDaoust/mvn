#! /usr/bin/env python
import numpy
import collections
import itertools
import functools
import re

import helpers            

from trydecorate import decorator

@decorator
def expandCallable(fun,self,other):
    return (
        fun(self,other(self.shape))
        if callable(other) else
        fun(self,other)
    )

class Matrix(numpy.matrix):
    """
    'Imporved' version of the martix class.
    the only modifications are:
        division (and rdiv) doesn't try to do elementwise division, it tries to 
            multiply by the inverse of the other
            
        The equality operator, ==, has also been modified to run numpy.allclose
        (good enough for me), so the matrix is treated as one thing, not 
        a collection of things.
            __eq__ accepts callables as arguments, like helpers.autostack, and 
            calls them with the matrixes size tuple as the only argument 
            
            >>> assert Matrix([[0,0],[0,0],[0,0]]) == numpy.zeros
            >>> assert Matrix([[1,0],[0,1]]) == Matrix.eye
    """
    rtol = 1e-5
    atol = 1e-8
    
    sign = helpers.sign

    def __new__(cls,data,dtype=None,copy=True):
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    @expandCallable
    def __eq__(self,other):
        other=Matrix(other)
        if self.shape==other.shape:
            return numpy.allclose(self,other,self.rtol,self.atol)
        else:
            ValueError('shape miss-match')

    def __ne__(self,other):
        return not(self ==  other)
    
    def __div__(self,other):
        return self*other**(-1)

    def __rdiv__(self,other):
        return other*self**(-1)

    @expandCallable
    def __add__(self,other):
        return numpy.matrix.__add__(self,other)
            
    def __repr__(self):
        return 'M'+numpy.matrix.__repr__(self)[1:]

    __str__ = __repr__

    def diagonal(self):
        return numpy.squeeze(numpy.array(numpy.matrix.diagonal(self)))
    
    def flatten(self):
        return numpy.array(self).flatten()

    def squeeze(self):
        return numpy.array(self).squeeze()

    @staticmethod
    def eye(*args,**kwargs):
        """
        improved version of numpy.eye
        
        behaves the same but will accept a shape tuple as a first 
        argument. 
        
        >>> assert Matrix.eye((2,2)) == Matrix.eye(2,2) == Matrix.eye(2)
        """
        #if isinstance(args[0],collections.Iterable):
        if hasattr(args[0],'__iter__'):
            args=itertools.chain(args[0],args[1:])
        return Matrix(numpy.eye(*args,**kwargs))

    @staticmethod
    def ones(shape,dtype=None,order='C'):
        return Matrix(numpy.ones(shape,dtype=None,order='C'))

    @staticmethod
    def zeros(shape,dtype=None,order='C'):
        return Matrix(numpy.zeros(shape,dtype=None,order='C'))

    @staticmethod
    def infs(shape,dtype=None,order='C'):
        return numpy.inf*Matrix.ones(shape,dtype=None,order='C')

    @staticmethod
    def nans(shape,dtype=None,order='C'):
        return numpy.nan*Matrix.ones(shape,dtype=None,order='C')

    @staticmethod
    def rand(shape):
        return numpy.random.rand(*shape)

    @staticmethod
    def randn(shape):
        return numpy.random.randn(*shape)
