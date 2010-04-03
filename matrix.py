import numpy
import collections

def eye(*args,**kwargs):
    """
    improved version of numpy.eye
    
    behaves the same but will accept a shape tuple as a first 
    argument. 
    
    >>> assert eye((2,2)) == eye(2,2) == eye(2)
    """
    if isinstance(args[0],collections.Iterable):
        args=itertools.chain(args[0],args[1:])
    return Matrix(numpy.eye(*args,**kwargs))

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
            >>> assert Matrix([[1,0],[0,1]]) == eye
            
            awsome eh?
    """
    def __new__(cls,data,dtype=None,copy=True):
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    def __eq__(self,other):
        return (
            numpy.allclose(self,other(self.shape)) 
            if callable(other) 
            else numpy.allclose(self,other)
        )
    
    def __div__(self,other):
        return self*other**(-1)

    def __rdiv__(self,other):
        return other*self**(-1)

    def __repr__(self):
        S=numpy.matrix.__repr__(self)
        return 'M'+S[1:]
    
    __str__ = __repr__

    def diagonal(self):
        return numpy.squeeze(numpy.array(numpy.matrix.diagonal(self)))
    