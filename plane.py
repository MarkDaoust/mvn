#! /usr/bin/env python

import numpy

from automath import automath
from right import right
from inplace import inplace

from matrix import Matrix 

@right
@inplace
@automath
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
        self.mean = numpy.real_if_close(stack[-1,1:])
        self.vectors = numpy.real_if_close(stack[:-1,1:])

def __and__(self,other):
        """
        plane intersection
        """
        assert 1==0, "this is just copied from mvar"

        #check if they both fill the space
        if (
            self.mean.size == (self.var!=0).sum() and 
            other.mean.size == (other.var!=0).sum()
        ):
            #then this is a standard paralell operation
            return (self**(-1)+other**(-1))**(-1) 
        
        #otherwise there is more work to do
        
        #inflate each object
        self=self.inflate()
        other=other.inflate()
        #collect the null vectors
        Nself=self.vectors[self.var==0,:]
        Nother=other.vectors[other.var==0,:] 

        #and stack them
        null=numpy.vstack([
            Nself,
            Nother,
        ])

        #get length of the component of the means along each null vector
        r=numpy.hstack([self.mean*Nself.H,other.mean*Nother.H])

        #square up the null vectors
        (s,v,d)=numpy.linalg.svd(null,full_matrices=False)

        #discard any very small components
        nonZero = ~helpers.approx(v**2)
        s=s[:,nonZero]
        v=v[nonZero]
        d=d[nonZero,:]
        
        #calculate the mean component in the direction of the new null vectors
        Dmean=r*s*numpy.diagflat(v**-1)*d
        
        #do the blending, while compensating for the mean of the working plane
        return ((self-Dmean)**-1+(other-Dmean)**-1)**-1+Dmean

