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

import helpers
from matrix import Matrix

def square(vectors,var=None,full=False):
    
    var =( 
        numpy.ones(vectors.shape[0]) if 
        var is None else 
        numpy.real_if_close(var)
    )

    infinite=helpers.approx(1/var**0.5)

    Ivar=numpy.array([])
    Ivectors=Matrix(numpy.zeros((0,vectors.shape[1])))

    if infinite.any():
        #square up the infinite vectors
        #Ivar is unused

        (Ivar,Ivectors)=_subSquare(vectors=vectors[infinite,:],var=numpy.ones_like(var[infinite]),full=True)
        Ivectors=Matrix(Ivectors)

        #take the finite variances and vectors
        var=var[~infinite]
        vectors=vectors[~infinite,:]

        Istd=abs(Ivar)**0.5
        
        small=helpers.approx(Istd)
        
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
        order=numpy.argsort(var)    
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

    if not numpy.all(shape):
        val=numpy.zeros([0])
        vec=numpy.zeros([0,shape[1]])
        return (val,vec)
    
    eig = (
        numpy.linalg.eigh if
        numpy.isreal(var).all() else
        numpy.linalg.eig
    )

    if shape[0]>=shape[1] or full or not vectors.any():
        scaled=Matrix(numpy.diagflat(var))*vectors
        
        cov=vectors.H*scaled
        (val,vec)=eig(cov)
        vec=vec.H

    elif not var.any():
        cov=vectors.H*vectors
        (_,vec)=eig(cov)
        vec=vec.H
        val=numpy.zeros(vec.shape[0])    

    else:
        scaled=Matrix(numpy.diagflat(var**(0.5+0j)))*vectors
        Xcov=vectors*vectors.H
        if Xcov == Matrix.eye:
            return (var,vectors)
        
        ( _ ,Xvec)=numpy.linalg.eig(Xcov)
        
        Xscaled=(Xvec.H*scaled)
        val=helpers.mag2(Xscaled)

        vec=numpy.array(Xscaled)/val[:,numpy.newaxis]**(0.5+0j)

    
    return (val,vec)

