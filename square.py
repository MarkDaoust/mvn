#!/usr/bin/env python
"""
this module contains one function: square

I was surprisedm when it worked. the idea was based on these two lines from wikipedia
 
http://en.wikipedia.org/wiki/Square_root_of_a_matrix:
(math notation converted to local python standard)
    '''if T = A*A.H = B*B.H, then there exists a unitary U s.t. 
    A = B*U'''

a unitary matrix is a complex rotation matrix
http://en.wikipedia.org/wiki/Unitary_matrix
    '''In mathematics, a unitary matrix is an nxn complex matrix U 
    satisfying the condition U.H*U = I, U*U.H = I'''
"""

import numpy
from matrix import Matrix
from helpers import ascomplex,mag2,autostack,squeeze,approx

def square(vectors,var,doSqueeze=True):

    #check the magnitudes of the variances
    infinite = approx(1/var**0.5)

    Ivar=numpy.array([])
    Ivectors=Matrix(numpy.zeros((0,vectors.shape[1])))

    if infinite.any():
        #square up the infinite vectors
        #Ivar is unused

        (Ivar,Ivectors)=_subSquare(vectors=vectors[infinite,:])

        #take the finite variances and vectors
        var=var[~infinite]
        vectors=vectors[infinite,:]

        assert isinstance(vectors,numpy.matrix)
        Ivectors=Matrix(Ivectors)

        (Ivar,Ivectors)=squeeze(vectors=Ivectors,var=Ivar)

        #revove the component paralell to each infinite vector
        vectors= vectors-(vectors*Ivectors.H)*Ivectors


    (var,vectors) = _subSquare(vectors,var)

    #if we want a real squeeze
    if doSqueeze:
        #do it
        (var,vectors)=squeeze(vectors=vectors,var=var)
    elif Ivar.size:
        #sort the finite variances
        order=numpy.argsort(var)    
        var=var(order)
        vectors=vectors[order,:]
        
        #if there are more vectors than dimensions 
        kill=var.size+Ivar.size-vectors.shape[1]
        if kill>0:
            #squeeze the vectors with the smallest variances 
            var=var[kill:]
            vectors=vectors[kill:,:]
    
    assert (var.size+Ivar.size-vectors.shape[1] <= 0),"""
        you should never have more vectors than dimensions
    """

    return (
        numpy.concatenate((var,numpy.inf*numpy.ones_like(Ivar))),
        numpy.vstack([vectors,Ivectors])
    )
    

def _subSquare(vectors,var=None):
    """
    given a series of vectors, this function calculates:
        (variances,vectors)=numpy.linalg.eigh(vectors.H*vectors)
    it's a seperate function because if there are less vectors 
    than dimensions the process can be accelerated, it just takes some dancing

    it is based on this:

    >>> vectors=Matrix(ascomplex(numpy.random.randn(
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

    var =( 
        numpy.ones(shape[0]) if 
        var is None else 
        numpy.real_if_close(var)
    )

    if not numpy.all(shape):
        val=numpy.zeros([0])
        vec=numpy.zeros([0,shape[1]])
        return (val,vec)

    varT=var[:,numpy.newaxis]
    
    eig = (
        numpy.linalg.eigh if
        Matrix(var) == abs(Matrix(var)) else
        numpy.linalg.eig
    )

    if shape[0]>=shape[1]:    
        scaled=Matrix(varT*numpy.array(vectors))
        
        cov=vectors.H*scaled
        (val,vec)=eig(cov)
        vec=vec.H
    else:    
        scaled=Matrix(varT**(0.5+0j)*numpy.array(vectors))
        Xcov=vectors*vectors.H
        if Xcov == Matrix.eye:
            return (var,vectors)
        
        ( _ ,Xvec)=numpy.linalg.eig(Xcov)
        
        Xscaled=(Xvec.H*scaled)
        val=mag2(Xscaled)

        vec=numpy.array(Xscaled)/val[:,numpy.newaxis]**(0.5+0j)

    
    return (val,vec)

