#! /usr/bin/env python
import numpy
from mvar import *

pickleName='testObjects.pkl'

def makeObjects(cplx=False,flat=False,ndim=None,seed=None):   
    rand=numpy.random.rand
    randn=numpy.random.randn
    randint=numpy.random.randint

    if seed is not None:
        numpy.random.seed(seed)

    if ndim is None:
        ndim=randint(1,20)

    if flat:
        ndim+=2
        num=randint(1,ndim-1)
    else:
        num=2*ndim
 
    #create n random vectors, 
    #with a default length of 'ndim', 
    #they can be made complex by setting cplx=True
    rvec= (lambda n=1:Matrix(helpers.ascomplex(randn(n,ndim,2)))) if cplx else (lambda n=1:Matrix(randn(n,ndim)))

    #create random test objects
    A=Mvar(
        mean=5*randn()*rvec(),
        vectors=5*randn()*rvec(num),
        #var=rand(num),
    )

    B=Mvar.fromCov(
        mean=5*randn()*rvec(),
        cov=(lambda x:x.H*x)(5*randn()*rvec(num))
    )

    C=Mvar.fromData(
        rvec(num+1)
    )

    A,B,C=numpy.random.permutation([A,B,C])
    
    M=rvec(ndim)
    M2=rvec(ndim)
    E=Matrix.eye(ndim)
    
    K1=randn()+randn()*(1j if cplx else 0)
    K2=randn()+randn()*(1j if cplx else 0)

    N=randint(-5,5)

    testObjects={
        'ndim':ndim,
        'A':A,'B':B,'C':C,
        'M':M,'M2':M2,'E':E,
        'K1':K1,'K2':K2,
        'N':N,
        'flat':flat,
        'cplx':cplx,
        'seed':seed,
    }

    return testObjects
