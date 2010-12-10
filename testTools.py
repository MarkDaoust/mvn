#! /usr/bin/env python
import numpy
from mvar import *

pickleName='testObjects.pkl'

def makeObjects(cplx=None,flat=None,ndim=None,seed=None):   
    rand=numpy.random.rand
    randn=numpy.random.randn
    randint=lambda x,y: numpy.round((x-0.5)+numpy.random.rand()*(y+0.5))

    if seed is not None:
        numpy.random.seed(seed)

    if ndim is None:
        ndim=randint(0,20)

    if flat is None:
        num=randint(0,2*ndim)
    elif flat:
        num=lambda :randint(0,ndim)
    else:
        num=lambda :2*ndim
 
    if cplx is None:
        cplx=numpy.rand
    else:
        c=bool(cplx)
        cplx=lambda:c

    #create n random vectors, 
    #with a default length of 'ndim', 
    #they can be made complex by setting cplx=True
    rvec= lambda n=1,ndim=ndim:Matrix(helpers.ascomplex(randn(n,ndim,2))) if cplx() else Matrix(randn(n,ndim))

    #create random test objects
    N=num()
    A=Mvar(
        mean=5*randn()*rvec(),
        vectors=5*randn()*rvec(N),
        var=rand(N),
    )

    B=Mvar.fromCov(
        mean=5*randn()*rvec(),
        cov=(lambda x:x.H*x)(5*randn()*rvec(num()))
    )

    C=Mvar.fromData(
        rvec(num()+1)
    )

    A,B,C=numpy.random.permutation([A,B,C])
    
    M=rvec(ndim)
    M2=rvec(ndim)
    E=Matrix.eye(ndim)
    
    K1=randn()+randn()*(1j if cplx() else 0)
    K2=randn()+randn()*(1j if cplx() else 0)

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
