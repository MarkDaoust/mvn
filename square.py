#!/usr/bin/env python
"""
this module contains one function: square
"""

import numpy
from matrix import Matrix
from operator import ge
from helpers import astype

def square(vectors):
    """
    given a series of vectors, this function calculates 
    """
    vectors=Matrix(vectors)
    shape=vectors.shape
    
    if not numpy.all(shape):
        var=numpy.zeros([0])
        vec=numpy.zeros([0,shape[1]])
    elif ge(*shape):
        cov=vectors.H*vectors
        (var,vec)=numpy.linalg.eigh(vectors.H*vectors)
        vec=vec.H
    else:
        Xcov=vectors*vectors.H
        
        (Xval,Xvec)=numpy.linalg.eigh(Xcov)
        
        var=numpy.diag(Xcov)
        
        vec=(Xvec.H*vectors).T
        vec=Matrix(((var**(-0.5+0j))*numpy.array(vec)).T)
        
    return (var,vec)

if __name__=='__main__':
    for n in xrange(1,20):
        shape=(numpy.random.randint(1,10),numpy.random.randint(1,10),2)
        vectors=Matrix(astype(numpy.random.randn(*shape),complex))
        
        (var,vec)=square(vectors)
        var=Matrix(numpy.diagflat(var))
        
        assert vec.H*(var)*vec==vectors.H*vectors
