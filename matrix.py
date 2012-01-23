#! /usr/bin/env python
"""
***************************
Modified Numpy Matrix Class
***************************
"""

#todo: update 'close' to match numpy's implementation for all_close  

import numpy
import itertools

import helpers            

from decorator import decorator

@decorator
def expandCallable(fun,self,other):
    """
    If the 'other' argument is a callable object call it with self.shape as the 
    only argument.
    """
    return (
        fun(self,other(self.shape))
        if callable(other) else
        fun(self,other)
    )

class Matrix(numpy.matrix):
    """
    'Imporved' version of the martix class.
            
    mostly a collection of minor tweaks and convienience functions.
    """
    
    rtol = 1e-5
    """
    Absolute tolerence for :py:meth:`mvar.matrix.Matrix.__eq__`
    
    passed as a parameter to :py:func:`numpy.allclose` to determine 'equality'
    """
    
    atol = 1e-8
    """
    Relative tolerence for :py:meth:`mvar.matrix.Matrix.__eq__`
    
    passed as a parameter to :py:func:`numpy.allclose` to determine 'equality'    
    """    
    
    sign = helpers.sign

    def __new__(cls,data,dtype=None,copy=True):
        """
        !!
        """
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    @expandCallable
    def __eq__(self,other):
        """
        Treats the matrix as a single object, and returns True or False.
        
        uses class members :py:attr:`mvar.matrix.Matrix.atol` and 
        :py:attr:`mvar.matrix.Matrix.rtol` through :py:func:`numpy.allclose` to 
        determine 'equality'
        
        Throws :py:class:`ValueError` if there is a shape miss-match 
        
        uses the :py:func:`mvar.matrix.expandCallable` decorator so that 
        this works
        
            >>> assert Matrix([[0,0],[0,0],[0,0]]) == numpy.zeros
            >>> assert Matrix([[1,0],[0,1]]) == Matrix.eye
        """
        other = Matrix(other)
        return numpy.allclose(self,other,self.rtol,self.atol)
        
    @expandCallable
    def __add__(self,other):
        """
        :py:func:`numpy.matrix.__add__` with the 
        :py:func:`mvar.matrix.expandCallable` decorator applied
        """
        return numpy.matrix.__add__(self,other)
        
    def __ne__(self,other):
        """
        inverse of __eq__
        
        return not (self == other)
        """
        return not (self ==  other)
    
    def __div__(self,other):
        """
        self/other == self*other**-1
        """
        return self*other**(-1)

    def __rdiv__(self,other):
        """
        other/self == other*self**-1
        """
        return other*self**(-1)
            
    def __repr__(self):
        return '\nM'+numpy.matrix.__repr__(self)[1:]

    __str__ = __repr__

    def diagonal(self):
        """
        return the diagonal of a matrix as a 1-D array
        see: :py:func:`numpy.diagonal`
        """
        return numpy.squeeze(numpy.array(numpy.matrix.diagonal(self)))
    
    def flatten(self):
        """
        copy the matrix to an array and flatten it
        see :py:func:`numpy.squeeze`
        """
        return numpy.array(self).flatten()

    def squeeze(self):
        """
        copy the matrix to an array and squeeze it
        see: :py:func:`numpy.squeeze`
        """
        return numpy.array(self).squeeze()
        
    def asarray(self):
        """
        return the data as an array
        see: :py:func:`numpy.asarray`
        """
        return numpy.asarray(self)
        
    def array(self):
        """
        return a copy of the data in an array
        see: :py:func:`numpy.array`
        """
        return numpy.array(self)

    @expandCallable        
    def approx(self,other = None):
        """
        same function as :py:func:`numpy.allclose`, but elementwise
        """
        if other is None :
            delta = numpy.abs(self)
            other = self
        else:
            delta = numpy.abs(self-other)
        
        return (delta < self.atol+self.rtol*numpy.abs(other))

    @classmethod
    def eye(cls,*args,**kwargs):
        """
        improved version of numpy.eye
        
        behaves the same but will accept a shape tuple as a first 
        argument. 
        
        >>> assert Matrix.eye((2,2)) == Matrix.eye(2,2) == Matrix.eye(2)
        
        see: :py:func:`numpy.eye`
        """
        #if isinstance(args[0],collections.Iterable):
        if hasattr(args[0],'__iter__'):
            args=itertools.chain(args[0],args[1:])
        return cls(numpy.eye(*args,**kwargs))

    @classmethod
    def ones(cls,shape=[],**kwargs):
        """
        return a matrix filled with ones
        see: :py:func:`numpy.ones`
        """
        return cls(numpy.ones(shape,**kwargs))

    @classmethod
    def zeros(cls,shape=[],**kwargs):
        """
        return a matrix filled with zeros
        see: :py:func:`numpy.zeros`
        """
        return cls(numpy.zeros(shape,**kwargs))

    @classmethod
    def infs(cls,shape=[],**kwargs):
        """
        return a matrix filled with infs
        """
        return numpy.inf*Matrix.ones(shape,**kwargs)

    @classmethod
    def nans(cls,shape=[],**kwargs):
        """
        return a matrix on filled with nans
        """
        return numpy.nan*Matrix.ones(shape,**kwargs)

    @classmethod
    def rand(cls,shape=[]):
        """
        return a matrix of uniformly distributed random numbers on [0,1]
        see: :py:func:`numpy.random.rand`        
        """
        return cls(numpy.random.rand(*shape))

    @classmethod
    def randn(cls,shape=[]):
        """
        return a matrix of normally distributed random numbers with unit variance
        see: :py:func:`numpy.random.randn`        
        """
        return cls(numpy.random.randn(*shape))
        
    @classmethod
    def stack(cls,rows,default = 0):
        """
        2d concatenation, expanding callables
        
        >>> E3 = numpy.eye(3)
        >>> Matrix.stack([ 
        ...     [         8*E3,Matrix.zeros],
        ...     [  Matrix.ones,           4],
        ... ])
        Matrix([[ 8.,  0.,  0.,  0.],
                [ 0.,  8.,  0.,  0.],
                [ 0.,  0.,  8.,  0.],
                [ 1.,  1.,  1.,  4.]])
        """
        return cls(helpers.autostack(rows,default))

    def det(self):
        """
        return the determinant of the matrix
        see: :py:func:`numpy.linalg.det`
        """
        return numpy.linalg.det(self)
        
        