#! /usr/bin/env python

#builtins
import os
import sys
import doctest
import cPickle

#3rd party
import numpy

#local
import mvar
import helpers
import square
import automath    
import right
import inplace
import matrix


from matrix import Matrix
from mvar import Mvar
from right import Right

#make a dictionary of the local modules
localMods={
    'mvar'    :mvar,
    'helpers' :helpers,
    'square'  :square,
#    'automath':automath,
#    'right'   :right,
#    'inplace' :inplace,
    'matrix'  :matrix,
}

pickleName='testObjects.pkl'
moduleName='testObjects.py'

def makeTestObjects(cplx=False,flat=False):   

    rand=numpy.random.rand
    randn=numpy.random.randn
    randint=numpy.random.randint

    if flat:
        ndim=3
        num=2
    else:
        ndim=2#randint(1,10)
        num=2*ndim
 
    #create n random vectors, 
    #with a default length of 'ndim', 
    #they can be made complex by setting cplx=True
    rvec=lambda n=1,m=ndim,cplx=cplx:Matrix(
        helpers.ascomplex(randn(n,m,2)) 
        if cplx else 
        randn(n,m)
    )

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

    C=Mvar.from_data(
        rvec(num+1)
    )

    A,B,C=numpy.random.permutation([A,B,C])
    
    M=rvec(ndim)
    M2=rvec(ndim)
    E=Matrix.eye(ndim)
    
    K1=randn()+randn()*1j
    K2=randn()+randn()*1j

    N=randint(-5,5)

    testObjects={
        'ndim':ndim,
        'A':A,'B':B,'C':C,
        'M':M,'M2':M2,'E':E,
        'K1':K1,'K2':K2,
        'N':N,
        'flat':flat    
    }

    return testObjects

def loadTestObjects():
    testObjects={}
    assert (
        'flat' not in sys.argv and 'cplx' not in sys.argv,
        "you can't set a type and reload at the same time"
    )

    print "#attempting to load pickle"        
    try:
        testObjects = cPickle.load(open(pickleName,'r'))
    except IOError:
        print "#    IOError"
    except  EOFError:
        print "#    EOFError"
    except cPickle.UnpicklingError:
        print "#    UnpicklingError"
    else:
        print "#loaded"
    
    return testObjects
    

def saveTestObjects(testObjects):
    pickleFile=open(pickleName,'w')

    print "#dumping new pickle"
    cPickle.dump(
        testObjects,
        pickleFile,
    )

testObjects={}

if '-r' in sys.argv:
    testObjects=loadTestObjects()

if not testObjects:
    testObjects=makeTestObjects(
        cplx='cplx' in sys.argv, 
        flat='flat' in sys.argv,
    )
    saveTestObjects(testObjects)

for name,mod in localMods.iteritems():
    mod.__dict__.update(testObjects)
    doctest.testmod(mod)

