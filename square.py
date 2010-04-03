#!/usr/bin/env python

import numpy
from matrix import Matrix
from operator import ge
from helpers import astype

def square(vectors):
    vectors=Matrix(vectors)
    
    if ge(*vectors.shape):
        cov=vectors.H*vectors
        (val,vec)=numpy.linalg.eigh(vectors.H*vectors)
        result=Matrix(val**(0.5+0j)*numpy.array(vec)).H
    else:
        Xcov=vectors*vectors.H
        (Xval,Xvec)=numpy.linalg.eigh(vectors*vectors.H)
        result=(Xvec.H*vectors)
        
    return result

def square2(vectors):
    vectors=Matrix(vectors)
    
    if ge(*vectors.shape):
        cov=vectors.H*vectors
        (val,vec)=numpy.linalg.eigh(vectors.H*vectors)
        val=val**(0.5+0j)
        vec=vec.H
    else:
        Xcov=vectors*vectors.H
        (Xval,Xvec)=numpy.linalg.eigh(vectors*vectors.H)
        
        val=numpy.diag(Xcov)
        val=val**(0.5+0j)
        
        vec=(Xvec.H*vectors).T
        vec=Matrix(((val**(-1))*numpy.array(vec)).T)
        
    return (val,vec)

if __name__=='__main__':
    for n in xrange(1,20):
        shape=(numpy.random.randint(1,10),numpy.random.randint(1,10),2)
        vectors=Matrix(astype(numpy.random.randn(*shape),complex))
        
        result=square(vectors);
        assert vectors.H*vectors==result.H*result,"the resulting vectros should have the same covariance as the input"
        (val,vec)=square2(vectors);
        val=Matrix(numpy.diagflat(val))
        
        assert vec.H*(val**2)vec==vectors.H*vectors
