import numpy
import collections
import itertools
import functools
import re

from helpers import sign

class Matrix(numpy.matrix):
    """
    Imporved version of the martix class.
    the only modifications are:
        division (and rdiv) doesn't try to do elementwise division, it tries to multiply 
            by the inverse of the other
            
        The equality operator, ==, has also been modified to run numpy.allclose
        (good enough for me), so the matrix is treated as one thing, not 
        a collection of things.
            __eq__ accepts callables as arguments, like helpers.autostack, and 
            calls them with the matrixes size tuple as the only argument 
            
            >>> assert Matrix([[0,0],[0,0],[0,0]]) == numpy.zeros
            >>> assert Matrix([[1,0],[0,1]]) == Matrix.eye
    """
    def __new__(cls,data,dtype=None,copy=True):
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    def __eq__(self,other):
        if callable(other):
            other=other(self.shape)
        
        other=Matrix(other)
        assert self.size==other.size,('can only compare two matrixes with the same size')
        return numpy.allclose(self,other)

    def __ne__(self,other):
        return not(self ==  other)
    
    def __div__(self,other):
        return self*other**(-1)

    def __rdiv__(self,other):
        return other*self**(-1)

    def __add__(self,other):
        if isinstance(other,numpy.matrix):
            assert self.shape == other.shape,'can only add matrixes with the same shape'
            return numpy.matrix.__add__(self,other)
        
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
    def ones(*args,**kwargs):
        return Matrix(numpy.ones(*args,**kwargs))

    @staticmethod
    def zeros(*args,**kwargs):
        return Matrix(numpy.zeros(*args,**kwargs))

    @staticmethod
    def infs(*args,**kwargs):
        return numpy.inf*Matrix.ones(*args,**kwargs)


    @staticmethod
    def nans(*args,**kwargs):
        return numpy.nan*Matrix.ones(*args,**kwargs)

Matrix.sign=sign
