#!/usr/bin/env python

import numpy
from matrix import Matrix
from operator import ge
from helpers import astype

def square(vectors):

    if ge(*vectors.shape):
        cov=vectors.H*vectors
        (val,vec)=numpy.linalg.eigh(vectors.H*vectors)
        vec=vec.H
        result=numpy.diagflat(val**(0.5+0j))*vec
    else:
        Xcov=vectors*vectors.H
        (Xval,Xvec)=numpy.linalg.eigh(vectors*vectors.H)
        result=(Xvec.H*vectors)
        
    return result

        
if __name__=='__main__':
    for n in xrange(1,20):
        shape=(numpy.random.randint(1,10),numpy.random.randint(1,10),2)
        vectors=Matrix(astype(numpy.random.rand(*shape),complex))

        
        result=square(vectors);
        assert vectors.H*vectors==result.H*result,"the resulting vectros should have the same covariance as the input"
        
