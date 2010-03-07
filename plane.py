import numpy

from helpers import autostack
from matrix import Matrix,eye

class Plane(object):
    """
    there's a problem here: this creates something who's transpose is it's unverse,
    not something who's conjugate transpose is it's inverse
    (and what do I even mean by inverse - they're usually not square) 
    
    >>> P1=Plane(vectors=Matrix(numpy.random.randn(3,3)))
    >>> assert P1.vectors.T*P1.vectors==(lambda shape:numpy.eye(*shape))
    
    >>> R=numpy.random.randn(3,3,2)
    >>> R.dtype=complex
    >>> P2=Plane(vectors=Matrix(R))
    >>> assert (P1.vectors.T*P1.vectors)!= eye,'this isn't supposed to match' 
    >>> assert (P1.vectors.H*P1.vectors)== eye,'this is'
    """ 
    
    def __init__(self,
        vectors=numpy.zeros,
        mean=numpy.zeros,
        do_square=True,
        do_unit=True
    ):
        stack=Matrix(autostack([
            [numpy.zeros,vectors],
            [          1,mean   ],
        ]))
        
        if len(vectors) > len(vectors[0]):
            raise ValueError('the sequence of vectors making up a plane must not be longer than the vectrors themselves')
        
        self.vectors=stack[:-1,1:]
        self.mean=stack[-1,1:]
        
        if do_square: self.vectors=square(vectors)
        if do_unit: self.vectors=unit(vectors)
        
    
def square(vectors):
    for (n,vector) in enumerate(vectors[:-1,:]):
        vectors[n+1:,:]-=project(vectors[n+1:,:],dir=vector)
        
    return vectors
    
def unit(vectors):
    return Matrix(numpy.array(vectors.T)/numpy.sqrt(numpy.sum(numpy.array(vectors)**2,1))).T
    
def project(vectors,dir):
    dir=unit(dir)
        
    return numpy.apply_along_axis(
        lambda vector,onto=dir: numpy.dot(vector,onto.T),        
        axis=1,
        arr=vectors,
    )*dir


if __name__=="__main__":
    import doctest
    doctest.testmod()

