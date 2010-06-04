import numpy
import collections
import itertools
import functools
import re

def sign(self):
    """
    improved sign function:
        returns a similar array of unit length (possibly complex) numbers pointing in the same 
        direction as the input 
    """
    return numpy.divide(
        self,
        abs(self),
    )

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
            
            awsome eh?
    """
    def __new__(cls,data,dtype=None,copy=True):
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    def __eq__(self,other):
        if callable(other):
            other=other(self.shape)

        if self.size==other.size:
            return numpy.allclose(self,other)
        else:
            raise ValueError('can only compare two matrixes with the same size')

    def __ne__(self,other):
        return not(self ==  other)
    
    def __div__(self,other):
        return self*other**(-1)

    def __rdiv__(self,other):
        return other*self**(-1)

    def __repr__(self):
        result='M'+numpy.matrix.__repr__(self)[1:]
        result=result.replace('[], shape=(','numpy.zeros(')
        for match in re.findall('numpy.zeros\([0-9, ]*\)',result):
            result=result.replace(
                match,
                match.replace('(','([').replace(')','])')
            )
        return result.replace('dtype=','dtype=numpy.')
    

    __str__ = __repr__

    def diagonal(self):
        return numpy.squeeze(numpy.array(numpy.matrix.diagonal(self)))
    
    def flatten(self):
        return numpy.array(self).flatten()

    def squeeze(self):
        return numpy.array(self).squeeze()

    @staticmethod
    @functools.wraps(numpy.eye)
    def eye(*args,**kwargs):
        """
        improved version of numpy.eye
        
        behaves the same but will accept a shape tuple as a first 
        argument. 
        
        >>> assert eye((2,2)) == eye(2,2) == eye(2)
        """
        #if isinstance(args[0],collections.Iterable):
        if hasattr(args[0],'__iter__'):
            args=itertools.chain(args[0],args[1:])
        return Matrix(numpy.eye(*args,**kwargs))

    @staticmethod
    @functools.wraps(numpy.ones)
    def infs(*args,**kwargs):
        return numpy.inf*numpy.ones(*args,**kwargs)


    @staticmethod
    @functools.wraps(numpy.ones)
    def nans(*args,**kwargs):
        return numpy.nan*numpy.ones(*args,**kwargs)

Matrix.sign=sign
