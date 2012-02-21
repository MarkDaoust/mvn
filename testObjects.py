#! /usr/bin/env python
"""
************
Test Objects
************
"""

import pickle
import numpy
import os

from mvn import Mvn
from mvn.matrix import Matrix

(dir,_) = os.path.split(os.path.abspath(__file__))

pickleName=os.path.join(dir,'testObjects.pkl')

testDict={}
try:
    testDict=pickle.load(open(pickleName,"r"))
    locals().update(testDict['last'])
except EOFError:
    pass
except IOError: 
    pass
except ValueError:
    pass
except KeyError:
    pass



def getObjects(values):
    filter = ['new','x','seed']
    frozenValues=frozenset(
        (key,value) 
        for (key,value) in values.__dict__.iteritems() 
        if key not in filter
    )

    objects=None
    if not values.new:
        try:
            objects=testDict[frozenValues]
        except KeyError:
            pass

    if objects is None:
        objects = makeObjects(values.flat,values.ndim,values.seed)
        
    testDict[frozenValues] = objects

    testDict['last']=objects
    globals().update(objects)

    pickle.dump(testDict,open(pickleName,'w'))

    return objects

def makeObjects(flat=None,ndim=None,seed=None):
    randn=numpy.random.randn
    randint=lambda x,y: int(numpy.round((x-0.5)+numpy.random.rand()*(y-x+0.5)))

    if seed is None:
        seed=randint(1,1e6)
    assert isinstance(seed,int)
    numpy.random.seed(seed)


    if ndim is None:
        ndim=randint(0,20)
    assert isinstance(ndim,int),'ndim must be an int'

    shapes={
        None:lambda :randint(-ndim,ndim),
        True:lambda :randint(1,ndim),
        False:lambda :randint(-ndim,0),
    }

    triple=lambda x:[x,x,x]
    
    if hasattr(flat,'__iter__'):
        flat=[f if isinstance(f,int) else shapes[f]() for f in flat]   
    elif flat in shapes:
        flat=[item() for item in triple(shapes[flat])]
    elif isinstance(flat,int):
        flat=triple(flat)
    
    assert all(f<=ndim for f in flat), "flatness can't be larger than ndim"
        

    rvec= lambda n=1,ndim=ndim:Matrix(randn(n,ndim))

    A,B,C=[
        Mvn.rand([ndim-F,ndim])
        for F in flat
    ]

# above we only get real varances, this would change them:
# ad kill all the unit tests
#
#   for V,D in zip([A,B,C],dtype):
#        V.var = V.var*(
#            numpy.random.randn(V.var.size)*D.real+
#            numpy.random.randn(V.var.size)*D.imag*1j
#        )

    n=randint(1,2*ndim)
    M=rvec(n).H
    M2=rvec(n).H    

    E=Matrix.eye(ndim)
    
    K1 = (numpy.random.randn())
    K2 = (numpy.random.randn())

    N=randint(-5,5)

    testDict={
        'ndim':ndim,
        'A':A,'B':B,'C':C,
        'M':M,'M2':M2,'E':E,
        'K1':K1,'K2':K2,
        'N':N,
        'seed':seed,
    }
    
    return testDict


