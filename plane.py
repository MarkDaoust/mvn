#! /usr/bin/env python

import numpy

import helpers
import decorate
from matrix import Matrix 
from square import square

@decorate.right
@decorate.inplace
@decorate.automath
class Plane(object):
    """
    plane class, meant to (eventually) factor out some code, and utility from the Mvar class
    """
    def __init__(
        self,
        vectors=Matrix.eye,
        mean=numpy.zeros,
    ):
        mean= mean if callable(mean) else numpy.array(mean).flatten()[None,:]
        vectors= vectors if callable(vectors) else Matrix(vectors)

        stack=Matrix(helpers.autostack([
            [vectors],
            [mean   ],
        ]))
        
        #unpack the stack into the object's parameters
        self.mean = numpy.real_if_close(stack[-1])
        self.vectors = numpy.real_if_close(stack[:-1])

    def __repr__(self):
        """
        print self
        """
        return '\n'.join([
            '%s(' % self.__class__.__name__,
            '    mean=',
            '        %s,' % self.mean.__repr__().replace('\n','\n'+8*' '),
            '    vectors=',
            '        %s' % self.vectors.__repr__().replace('\n','\n'+8*' '),
            ')',
        ])

    __str__ = __repr__

    @decorate.prop
    class shape():
        """
        get the shape of the vectors,the first element is the number of 
        vectors, the second is their lengths: the number of dimensions of 
        the space they are embedded in
            
        >>> assert A.vectors.shape == A.shape
        >>> assert (A.var.size,A.mean.size)==A.shape
        >>> assert A.shape[0]==A.rank
        >>> assert A.shape[1]==A.ndim
        """
        def fget(self):
            return self.vectors.shape

    def __pow__(self,other):
        assert other==-1
        return ~self

    def __invert__(self):
        """
        switch between vectors along the plane and vectors across the plane
        
        """
        shape=self.shape        

        result = self.copy()

        missing = shape[1]-shape[0]

        result.vectors = numpy.vstack(
            [self.vectors,numpy.zeros((missing,shape[1]))]
        )

        var,vectors = square(result.vectors)

        zeros=helpers.approx(var)

        result.vectors = vectors[zeros]

        return result

    def __add__(self,other):
        return Plane(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors])
        )

    def square(self):
        pass

    def __and__(self,other):
        """
        plane intersection
        """
        return (self**(-1)+other**(-1))**(-1) 

