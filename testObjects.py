import numpy
import pickle


from matrix import Matrix
from mvar import Mvar
import helpers

pickleName='testObjects.pkl'

def makeObjects(cplx=None,flat=None,ndim=None,seed=None):   
    rand=numpy.random.rand
    randn=numpy.random.randn
    randint=lambda x,y: int(numpy.round((x-0.5)+numpy.random.rand()*(y-x+0.5)))

    if seed is None:
        seed=randint(1,1e9)
    
    numpy.random.seed(seed)

    if ndim is None:
        ndim=randint(0,20)

    if flat is None:
        num=lambda :randint(0,2*ndim)
    elif flat:
        num=lambda :randint(0,ndim)
    else:
        num=lambda :randint(ndim+1,2*ndim)
 
    if cplx is None:
        cplx=lambda :numpy.round(numpy.random.rand())
    else:
        c=bool(cplx)
        cplx=lambda:c

    #create n random vectors, 
    #with a default length of 'ndim', 
    #they can be made complex by setting cplx=True
    rvec= lambda n=1,ndim=ndim:Matrix(helpers.ascomplex(randn(n,ndim,2))) if cplx() else Matrix(randn(n,ndim))

    #create random test objects
    A=Mvar(
        mean=5*randn()*rvec(),
        vectors=5*randn()*rvec(num()),
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

