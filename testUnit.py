#! /usr/bin/env python

import unittest
import testObjects
from testObjects import *
from mvar import *

class myTest(unittest.TestCase):
    def setup(self):
        testObjects = reload(testObjects)
        globals().update(testObjects.__dict__)

    def testEq(self):
        assert A==A
        assert A!=B
        assert A==A.square()
        assert A==A.copy()
        assert A==A.inflate()
        assert A==A.inflate().squeeze()

    def testPlus(self):
        assert A == +A 
        assert A == ++A
        assert A is not +A
        assert A+A == 2*A
        assert A+A+A == 3*A

    def testMinus(self):
        assert A == --A
        assert A--B==A+B
        assert -A == -1*A
        assert A-B == A+(-B)

    def testMul(self):
        assert A**2==A*A

    def testDiv(self):
        assert A/B == A*B**(-1)

    def testCopy(self):
        A2=A.copy(deep=True)        
        assert A2 == A
        assert A2 is not A
        assert A2.mean is not A.mean

        A.copy(B,deep=True)
        assert B == A
        assert B is not A
        assert A.mean is not B.mean

        A2=A.copy(deep=False)        
        assert A2 == A
        assert A2 is not A
        assert A2.mean is A.mean

        A.copy(B,deep=False)
        assert B == A
        assert B is not A
        assert A.mean is B.mean

    def testNdim(self):
        assert A.ndim == A.mean.size

    def testShape(self):
        assert A.vectors.shape == A.shape
        assert (A.var.size,A.mean.size)==A.shape
        assert A.shape[1]==A.ndim


    def testCov(self):
        assert A.vectors.H*numpy.diagflat(A.var)*A.vectors == A.cov
        assert A.transform()**2 == abs(A).cov 
        assert A.transform(2) == abs(A).cov
        if not(flat and N<0):
            assert A.transform()**N == A.transform(N)

    def testScaled(self):
        assert A.scaled.H*A.scaled==abs(A).cov
        assert Matrix(helpers.mag2(A.scaled))==A.var
        assert A.vectors.H*A.scaled==A.transform()

    def testFromData(self):
        assert Mvar.fromData(A)==A 

        data=[1,2,3]
        new=Mvar.fromData(data)
        assert new.mean == data
        assert (new.var == numpy.zeros([0])).all()
        assert new.vectors == numpy.zeros
        assert new.cov == numpy.zeros([3,3])

    def testZeros(self):
        n=abs(N)
        Z=Mvar.zeros(n)
        assert Z.mean==Matrix.zeros
        assert Z.var.size==0
        assert Z.vectors.size==0

    def testInfs(self):
        n=abs(N)
        inf=Mvar.infs(n)
        assert inf.mean==Matrix.zeros
        assert inf.var.size==inf.mean.size==n
        assert (inf.var==numpy.inf).all()
        assert inf.vectors==Matrix.eye


    def testVectors(self):
        if A.shape[0] == A.shape[1]:
            assert A.vectors.H*A.vectors==Matrix.eye
        else:
            a=A.inflate()
            assert a.vectors.H*a.vectors==Matrix.eye
        
    def testTransform(self):
        assert A.transform() == A.transform(1)
        assert A.transform() == A.scaled.H*A.vectors
    
    def testCov(self):
        assert A.cov == (A**2).transform()
        assert A.cov == A.transform()*A.transform()
        assert A.cov == A.transform()**2
        assert A.cov == A.transform(2)
        assert (A*B.transform() + B*A.transform()).cov/2 == (A*B).cov

    def testIntPowers(self):
        assert A.transform(N)== (A**N).transform()
        assert A.transform(N)== A.transform()**N 

    def testRealPowers(self):
        assert (A**K1.real).transform() == A.transform(K1.real) 

    def testComplexPowers(self):
        assert (A**K1).transform() == A.transform(K1) 

    def testTrace(self):
        assert Matrix(numpy.trace(A.transform(0))) == A.shape[0] 
        assert Matrix(A.trace()) == A.var.sum()
        assert Matrix(A.trace()) == numpy.trace(A.cov)

    def testDet(self):
        assert Matrix(A.det())== numpy.linalg.det(A.cov)
        assert Matrix(A.det()) == (
             0 if 
             A.shape[0]!=A.shape[1] else 
             numpy.prod(A.var)
        )

    def testDist2(self):
        if not flat:
            assert Matrix((A**0).dist2(numpy.zeros((1,ndim))))==helpers.mag2((A**0).mean)
            
    def testGivenScalar(self):
        a = A.given(index=0,value=1)
        assert a.mean[:,0]==1
        assert a.vectors[:,0]==numpy.zeros

        a=A.copy(deep=True)
        a[0]=1
        assert a==A.given(index=0,value=1)


    def testGivenLinear(self):
        L1=Mvar(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
        L2=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf) 
        assert L1.given(index=0,value=1) == L1&L2
        assert (L1&L2).mean==[1,1]
        assert (L1&L2).cov==[[0,0],[0,2]]

    def testGivenMvar(self):
        Y=Mvar(mean=[0,1],vectors=Matrix.eye, var=[numpy.inf,1])
        X=Mvar(mean=[1,0],vectors=Matrix.eye,var=[1,numpy.inf])
        x=Mvar(mean=1,var=1)
        assert Y.given(index=0,value=x) == X&Y

    def testMGiven(self):
        assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)


    def testEq(self):
        assert A==A.copy()
        assert A is not A.copy()
        assert A != B

        assert (
             Mvar(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
             Mvar(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
        )

    def testInvert(self):
        assert (A.var>=0).all()
        assert abs(A) == abs(~A)

        IA=A.copy(deep=True)
        IA.var*=-1
        assert IA == ~A

        assert (~A).mean == A.mean

        assert Matrix((~A).var) == (-A).var 
        assert Matrix((~A).var) == -(A.var)

        assert (~A).vectors==A.vectors

        assert (~A).cov == (-A).cov 
        assert (~A).cov == -(A.cov)

        assert ~~A==A

        assert ~(~A&~B) == A & B 

        assert (A & ~A) == Mvar(mean=A.mean, vectors=A.vectors, var=Matrix.infs)

        if not flat:
            assert (A & ~A) == Mvar(mean=numpy.zeros(A.ndim))**-1
            assert A == A & (B & ~B)
            assert (A&B) & ~B == A & (B&~B)

        assert  A &(B & ~B) == A & Mvar(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
        assert (A&~B) & B == (A&B) & ~B

        assert not numpy.isfinite((A & ~A).var).any()

        P=A.copy()
        P.var=P.var/0.0
        assert P==(A & ~A)       

        assert (~A & ~B) == ~(A & B)

    def testBlend(self):
        assert A & B == B & A 
        
        assert (A & A).cov == A.cov/2
        assert (A & A).mean == A.mean
        
        if not flat:
            assert A & B == 1/(1/A+1/B)
            assert A &-A == Mvar(mean=numpy.zeros(ndim))**-1
            assert A &~A == Mvar(mean=numpy.zeros(ndim))**-1
            assert A & B == wiki(A,B)
            
            abc=numpy.random.permutation([A,B,C])
            assert A & B & C == helpers.paralell(*abc) or flat
            assert A & B & C == reduce(operator.and_ ,abc) or flat
    
            assert (A & B) & C == A & (B & C) or flat


        L1=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf)
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        assert (L1&L2).mean==[1,1]
        assert (L1&L2).var.size==0

        L1=Mvar(mean=[0,0],vectors=[1,1],var=numpy.inf)
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        assert (L1&L2).mean==[1,1]
        assert (L1&L2).var.size==0

        L1=Mvar(mean=[0,0],vectors=Matrix.eye, var=[1,1])
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        assert (L1&L2).mean==[0,1]
        assert (L1&L2).var==1
        assert (L1&L2).vectors==[1,0]

        if not flat:    
             assert A**-1 == A*A**-2 or flat
             assert A & B == (A*A**-2+B*B**-2)**-1 or flat

             D = A*(A.cov)**(-1) + B*(B.cov)**(-1)
             assert wiki(A,B) == D*(D.cov)**(-1)
             assert A & B == wiki(A,B)

    def testZeroPow(self):
        assert A**0*A==A
        assert A*A**0==A
        if not flat:
            assert A**0 == A**(-1)*A
            assert A**0 == A*A**(-1)
            assert A**0 == A/A 
            assert (A**0).mean == A.mean*(A**-1).transform()
            assert (A**0).mean == A.mean*A.transform(-1)

        assert Matrix((A**0).var) == numpy.ones

    def testOnePow(self):
        assert A==A**1
        assert -A == (-A)**1

        if not flat:
            assert A == (A**-1)**-1

    def testRealPow(self):
        assert A*A==A**2
        assert A/A**-1 == A**2
        
        if not flat:
            k=K1.real
            assert A**k == A*A.transform(k-1) + Mvar(mean=A.mean-A.mean*A.transform(0)) 
    
        assert A.mean*A.transform(0) == ((A**-1)**-1).mean

    
        assert (A**K1.real)*(A**K2.real)==A**(K1.real+K2.real)
        assert A**K1.real/A**K2.real==A**(K1-K2)

    def testMulTypes(self):
        assert isinstance(A*B,Mvar)
        assert isinstance(A*M,Mvar)
        assert isinstance(M*A,Matrix) 
        assert isinstance(A*K1,Mvar)
        assert isinstance(K1*A,Mvar)

    def testMvarMul(self):
        assert (A*B).cov == (A*B.transform()+B*A.transform()).cov/2
        assert (A*B).mean == (A*B.transform()+B*A.transform()).mean/2 or flat

        assert M*B==M*B.transform()
        if not flat:
            assert A**2==A*A.transform() or flat

        assert A*(B**0+B**0) == A*(2*B**0)
        assert (A*B**0 + A*B**0).cov == (2*A*B**0).cov 

        assert A*A==A**2
        assert A*A==A*A.transform() or flat
        assert A*B == B*A


    def testScalarMul(self):
        assert A+A == 2*A
        assert (2*A).mean==2*A.mean
        assert (2*A.cov) == 2*A.cov
    
        assert A*(K1+K2)==A*K1+A*K2

        assert (A*(K1*E)).mean == K1*A.mean
        assert (A*(K1*E)).cov == A.cov*abs(K1)**2
        assert (K1*A)*K2 == K1*(A*K2)

        assert (2*B**0).transform() == sqrt(2)*(B**0).transform()    
        
        assert A*K1 == K1*A

        assert (A*K1).mean==K1*A.mean
        assert (A*K1).cov== (A.cov)*K1

        assert (A*(E*K1)).mean==A.mean*K1
        assert (A*(E*K1)).cov ==(E*K1).H*A.cov*(E*K1)


    def testMixedMul(self):
        assert K1*A*M == A*K1*M 
        assert K1*A*M == A*M*K1

    def testMatrixMul(self):
        assert (A*M)*M2 == A*(M*M2)
        assert (M*M2)*A == M*(M2*A)

        assert (A*M).cov == M.H*A.cov*M
        assert (A**2).transform() == A.cov
        
        assert A*E==A
        assert (-A)*E==-A 

        assert (A*M).cov==M.H*A.cov*M
        assert (A*M).mean==A.mean*M

    def testDiv(self):
        assert A/B == A*(B**(-1))
        assert A/M == A*(M**(-1))
        assert A/K1 == A*(K1**(-1))
        assert K1/A == K1*(A**(-1))
        assert M/A==M*(A**(-1))

    def testPos(self):
        assert A==+A==++A

    def testAdd(self):
        assert (A+A)==(2*A)

        assert (A+A).mean==(2*A).mean
        assert (A+A).mean==2*A.mean

        assert sum(itertools.repeat(A,N-1),A) == A*(N)

        assert (A+B)==Mvar(
            mean=A.mean+B.mean,
            vectors=numpy.vstack([A.vectors,B.vectors]),
            var = numpy.concatenate([A.var,B.var]),
        )

        assert (A+A).mean==(2*A).mean
        assert (A+A).mean==2*A.mean

        assert (A+B).mean==A.mean+B.mean
        assert (A+B).cov==A.cov+B.cov


    def testNeg(self):
        assert -A==-1*A
        assert A==-1*-1*A
        assert 1j*A*1j==1j*(1j*A) ==-A

    def testSub(self):
        assert B+(-A) == B+(-1)*A == B-A    
        assert (B-A)+A==B

        assert A-A == Mvar(mean=numpy.zeros_like(A.mean))
        assert (A-B)+B == A
        assert (A-B).mean == A.mean - B.mean
        assert (A-B).cov== A.cov - B.cov

        assert (A+B*(-E)).mean == A.mean - B.mean
        assert (A+B*(-E)).cov== A.cov + B.cov

        assert A-B == -(B-A)
        assert A+(B-B)==A


    def testRight(self):
        assert A+B == B+A
        assert A*B == B*A
        assert B-A == B+(-A)
        assert B/A == B*A**(-1)
        assert A & B == B & A
        assert A | B == B | A
        assert A ^ B == B ^ A

    def testSquare(myTest):
        vectors=Matrix(helpers.ascomplex(numpy.random.randn(
            numpy.random.randint(1,10),numpy.random.randint(1,10),2
        )))
        cov = vectors.H*vectors
        Xcov = vectors*vectors.H 
        (Xval,Xvec) = numpy.linalg.eigh(Xcov)
        vec = Xvec.H*vectors
        assert vec.H*vec == cov

if __name__ == '__main__':
    unittest.main()
