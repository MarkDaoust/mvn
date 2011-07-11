#!/usr/bin/env python
"""
This module contains one function: square
"""

import numpy
import scipy

import helpers
from matrix import Matrix

def square(vectors,var=None,full=False):
    """
    calculates the eigen-vectors and eigen-values of the covariance matrix that 
    would be produced by multiplying out A.var*numpy.array(A.vectors.H)*A.vectors 
    without necessarily calculating the covariance matrix itself.

    It is also setup to handle vectors with infinite variances.
    origionally the idea came from these two line on wikipedia:

    http://en.wikipedia.org/wiki/Square_root_of_a_matrix:
        '''if T = A*A.H = B*B.H, then there exists a unitary U s.t. A = B*U'''

    http://en.wikipedia.org/wiki/Unitary_matrix
        '''In mathematics, a unitary matrix is an nxn complex matrix U satisfying 
        the condition U.H*U = I, U*U.H = I'''

    *********************************
    A better description for all this is the compact singular value decomposition.
    http://en.wikipedia.org/wiki/Singular_value_decomposition#Compact_SVD

    but here I only need one of the two sets of vectors, so I actually calculate the smaller of 
    the two possible covariance marixes and, and then it's eigen-stuff.
    """ 
    if var is None:
        var =numpy.ones(vectors.shape[0]) 

    infinite=helpers.approx(1/scipy.sqrt(var)) | ~numpy.isfinite(var)

    Ivar=numpy.array([])
    Ivectors=Matrix(numpy.zeros((0,vectors.shape[1])))

    if infinite.any():
        #square up the infinite vectors
        #Ivar is unused

        (Ivar,Ivectors)=_subSquare(vectors=vectors[infinite,:],var=numpy.ones_like(var[infinite]),full=True)

        #take the finite variances and vectors
        var=var[~infinite]
        vectors=vectors[~infinite,:]
        
        small=helpers.approx(Ivar)
        
        Ivar = Ivar[~small]

        SIvectors = Ivectors[~small,:]

        if vectors.any():
            #revove the component paralell to each infinite vector
            vectors= vectors-vectors*SIvectors.H*SIvectors            
        elif var.size :
            num= helpers.approx(var).sum()
            #gab the extra vectors here, because if the vectors are all zeros eig will fail
            vectors=Ivectors[small,:]
            vectors=vectors[:num,:]

        Ivectors=SIvectors
        
    if var.size:
        (var,vectors) = _subSquare(vectors,var)

    if Ivar.size and var.size:
        #sort the finite variances
        order=numpy.argsort(abs(var))    
        var=var[order]
        vectors=vectors[order,:]
        
        #if there are more vectors than dimensions 
        kill=var.size+Ivar.size-vectors.shape[1]
        if kill>0:
            #squeeze the vectors with the smallest variances 
            var=var[kill:]
            vectors=vectors[kill:,:]
    
    return (
        numpy.concatenate((var,numpy.inf*numpy.ones_like(Ivar))),
        numpy.vstack([vectors,Ivectors])
    )
    

def _subSquare(vectors,var,full=False):
    """
    given a series of vectors, this function calculates:
        (variances,vectors)=numpy.linalg.eigh(vectors.H*vectors)
    it's a seperate function because if there are less vectors 
    than dimensions the process can be accelerated, it just takes some dancing

    it is based on this:

    >>> vectors=Matrix(helpers.ascomplex(numpy.random.randn(
    ...     numpy.random.randint(1,10),numpy.random.randint(1,10),2
    ... )))
    >>> cov = vectors.H*vectors
    >>> Xcov = vectors*vectors.H 
    >>> (Xval,Xvec) = numpy.linalg.eigh(Xcov)
    >>> vec = Xvec.H*vectors
    >>> assert vec.H*vec == cov
    """
    vectors=Matrix(vectors)
    shape=vectors.shape

    if not all(shape):
        val=numpy.zeros([0])
        vec=numpy.zeros([0,shape[1]])
        return (val,vec)
    
    eig = numpy.linalg.eigh

    if shape[0]>=shape[1] or full or not vectors.any() or (var<0).any():
        scaled=Matrix(var[:,None]*numpy.array(vectors))
        
        cov=vectors.H*scaled
        (val,vec)=eig(cov)
        vec=vec.H

    elif not var.any():
        cov=vectors.H*vectors
        (_,vec)=eig(cov)
        vec=vec.H
        val=numpy.zeros(vec.shape[0])

    else:
        scaled=Matrix(scipy.sqrt(var)[:,None]*numpy.array(vectors))
        #Xcov = scaled*scaled.H        
        Xcov=var[:,None]*numpy.array(vectors)*vectors.H
        
        (_,Xvec)=eig(Xcov)
        
        Xscaled=(Xvec.H*scaled)
        val=helpers.mag2(Xscaled)

        vec=numpy.array(Xscaled)/scipy.sqrt(val[:,numpy.newaxis])

    
    return (val,vec)

