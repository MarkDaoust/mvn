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
    plane class, meant to factor out some code, and utility from the Mvar class
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
