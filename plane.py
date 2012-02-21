#! /usr/bin/env python
"""
****************
Plane Base Class
****************
"""
import numpy

import mvn.helpers as helpers
import mvn.decorate as decorate
from mvn.matrix import Matrix 
from mvn.square import square

def getNull(vectors):
    shape=vectors.shape        
    missing = shape[1]-shape[0]

    if missing>0:
        vectors = numpy.vstack(
            [vectors,numpy.zeros((missing,shape[1]))]
        )
    else:
        vectors=vectors

    var,vectors = square(vectors)

    zeros=helpers.approx(var)

    return vectors[zeros]


Plane = decorate.underConstruction('Plane')

@decorate.right
@decorate.inplace
@decorate.automath
@decorate.MultiMethod.sign(Plane)
class Plane(object):
    """
    plane class, meant to (eventually) factor out some code, and utility from the Mvn class
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
        ],default=1))
        
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

    @decorate.MultiMethod
    def __add__(self,other):
        """
        add two planes together
        """
        raise TypeError("No Apropriate Method Found")

    @__add__.register(Plane)
    def __add__(self,other):
        result = self.copy()
        result.mean = result.mean+other
        return result

    @__add__.register(Plane,Plane)
    def __add__(self,other):
        return Plane(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors])
        )

    def getNull(self):
        return getNull(self.vectors)


    def __and__(self,other):
        """
        plane intersection
        """
        Nself=self.getNull()
        Nother=other.getNull()

        #and stack them
        null=numpy.vstack([
            Nself,
            Nother,
        ])

        #get length of the component of the means along each null vector
        r=numpy.vstack([Nself*self.mean.H,Nother*other.mean.H])
        
        mean = (numpy.linalg.pinv(null,1e-6)*r).H

        return Plane(vectors=getNull(null),mean=mean)


