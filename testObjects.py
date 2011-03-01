
import random
import pickle
import numpy

from __init__ import Mvar
from matrix import Matrix
import helpers

pickleName='testObjects.pkl'

def makeObjects(dtype=None,flat=None,ndim=None,seed=None):
    rand=numpy.random.rand
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
        flat=triple(x)
    
    assert all(f<=ndim for f in flat), "flatness can't be larger than ndim"


    dtypes={
        None:lambda:[1.0,1.0+1.0j,1.0j][randint(0,2)],
        'r':lambda:1.0+0j,
        'c':lambda:(1.0+1.0j)/(2**0.5),
        'i':lambda:0+1.0j,
    }
 
    if hasattr(dtype,'__iter__'):
        dtype=[
                complex(item) if 
                isinstance(item,(float,complex,int)) else 
                dtypes[item]() 
            for item in dtype
        ]
    elif dtype in dtypes:
        dtype=[item() for item in triple(dtypes[dtype])]
        
    rvec= lambda n=1,dtype=1+0j,ndim=ndim:Matrix(
        randn(n,ndim)*dtype.real+
        randn(n,ndim)*dtype.imag
    )

    [A,B,C]=[
        Mvar(
            mean=5*randn()*rvec(dtype=D),
            vectors=5*randn()*rvec(n=ndim-F,dtype=D),
        ) for F,D in zip(flat,dtype)
    ]
 
    
    M=rvec(n=ndim,dtype=random.choice(dtype))
    M2=rvec(n=ndim,dtype=random.choice(dtype))
    E=Matrix.eye(ndim)
    
    K1 = (numpy.random.randn()+numpy.random.randn()*1j)
    K2 = (numpy.random.randn()+numpy.random.randn()*1j)

    N=randint(-3,3)

    testObjects={
        'ndim':ndim,
        'A':A,'B':B,'C':C,
        'M':M,'M2':M2,'E':E,
        'K1':K1,'K2':K2,
        'N':N,
        'seed':seed,
    }

    
    pickle.dump(testObjects,open(pickleName,'w'))

    return testObjects

try:
    testObjects=pickle.load(open(pickleName,"r"))
except EOFError:
    pass
except IOError:
    pass
except ValueError:
    pass
else:
    locals().update(testObjects)




#    #create random test objects
#    A=Mvar(
#        mean=5*randn()*rvec(),
#        vectors=5*randn()*rvec(num()),
#    )
#    B=Mvar.fromCov(
#        mean=5*randn()*rvec(),
#        cov=(lambda x:x.H*x)(5*randn()*rvec(num()))
#    )
#    C=Mvar.fromData(
#        rvec(num()+1)
#    )
#   A,B,C=numpy.random.permutation([A,B,C])
