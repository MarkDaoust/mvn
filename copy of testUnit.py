#! /usr/bin/env python

import unittest
import testObjects
import cPickle
from mvar import *

jar=open('testObjects.pkl').read()

class myTests(unittest.TestCase):
    def setUp(self):
       self.__dict__.update(cPickle.loads(jar))


class commuteTester(myTests):
   def testRight(self):
        assert self.A+self.B == self.B+self.A
        assert self.A*self.B == self.B*self.A
        assert self.B-self.A == (-self.A)+self.B
        assert self.A/self.B == (self.B**-1)*self.A
        assert self.A & self.B == self.B & self.A
        assert self.A | self.B == self.B | self.A
        assert self.A ^ self.B == self.B ^ self.A

class creationTester(myTests):
    def testFromData(self):
        assert Mvar.fromData(self.A)==self.A 

        data=[1,2,3]
        new=Mvar.fromData(data)
        assert new.mean == data
        assert (new.var == numpy.zeros([0])).all()
        assert new.vectors == numpy.zeros
        assert new.cov == numpy.zeros([3,3])

    def testZeros(self):
        n=abs(self.N)
        Z=Mvar.zeros(n)
        assert Z.mean==Matrix.zeros
        assert Z.var.size==0
        assert Z.vectors.size==0

    def testInfs(self):
        n=abs(self.N)
        inf=Mvar.infs(n)
        assert inf.mean==Matrix.zeros
        assert inf.var.size==inf.mean.size==n
        assert (inf.var==numpy.inf).all()
        assert inf.vectors==Matrix.eye

    def testCopy(self):
        self.A2=self.A.copy(deep=True)        
        assert self.A2 == self.A
        assert self.A2 is not self.A
        assert self.A2.mean is not self.A.mean

        self.A.copy(self.B,deep=True)
        assert self.B == self.A
        assert self.B is not self.A
        assert self.A.mean is not self.B.mean

        self.A2=self.A.copy(deep=False)        
        assert self.A2 == self.A
        assert self.A2 is not self.A
        assert self.A2.mean is self.A.mean

        self.A.copy(self.B,deep=False)
        assert self.B == self.A
        assert self.B is not self.A
        assert self.A.mean is self.B.mean


class equalityTester(myTests):
    def testEq(self):
        # always equal if same object
        assert self.A==self.A
        # still equal if copy
        assert self.A==self.A.copy()
        assert self.A is not self.A.copy()
    
    def testCosmetic(self):
        assert self.A==self.A.square()
        assert self.A==self.A.copy()
        assert self.A==self.A.inflate()
        assert self.A==self.A.inflate().squeeze()

    def testInf(self):
        assert (
             Mvar(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
             Mvar(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
        )
        
    def testNot(self):
        assert self.A!=self.B

class signTester(myTests):
    def testPlus(self):
        assert self.A == +self.A 
        assert self.A == ++self.A
        assert self.A is not +self.A
        assert self.A+self.A == 2*self.A
        assert self.A+self.A+self.A == 3*self.A

    def testMinus(self):
        assert -self.A == -1*self.A
        assert self.A == --self.A
        assert self.A--self.B==self.A+self.B
        assert self.A-self.B == self.A+(-self.B)

    def testPos(self):
        assert self.A==+self.A==++self.A
    
    def testNeg(self):
        assert -self.A==-1*self.A
        assert self.A==-1*-1*self.A
        assert 1j*self.A*1j==1j*(1j*self.A) ==-self.A

    def testsAdd(self):
        assert (self.A+self.A)==(2*self.A)

        assert (self.A+self.A).mean==(2*self.A).mean
        assert (self.A+self.A).mean==2*self.A.mean

        n=abs(self.N)
        assert numpy.array(itertools.repeat(self.A,n)).sum() == self.A*n

        assert (self.A+self.B)==Mvar(
            mean=self.A.mean+self.B.mean,
            vectors=numpy.vstack([self.A.vectors,self.B.vectors]),
            var = numpy.concatenate([self.A.var,self.B.var]),
        )

        assert (self.A+self.A).mean==(2*self.A).mean
        assert (self.A+self.A).mean==2*self.A.mean

        assert (self.A+self.B).mean==self.A.mean+self.B.mean
        assert (self.A+self.B).cov==self.A.cov+self.B.cov

    def testSub(self):
        assert self.B+(-self.A) == self.B+(-1)*self.A == self.B-self.A    
        assert (self.B-self.A)+self.A==self.B

        assert self.A-self.A == Mvar(mean=numpy.zeros_like(self.A.mean))
        assert (self.A-self.B)+self.B == self.A
        assert (self.A-self.B).mean == self.A.mean - self.B.mean
        assert (self.A-self.B).cov== self.A.cov - self.B.cov

        assert (self.A+self.B*(-self.E)).mean == self.A.mean - self.B.mean
        assert (self.A+self.B*(-self.E)).cov== self.A.cov + self.B.cov

        assert self.A-self.B == -(self.B-self.A)
        assert self.A+(self.B-self.B)==self.A

class productTester(myTests):
    def testMulTypes(self):
        assert isinstance(self.A*self.B,Mvar)
        assert isinstance(self.A*self.M,Mvar)
        assert isinstance(self.M*self.A,Matrix) 
        assert isinstance(self.A*self.K1,Mvar)
        assert isinstance(self.K1*self.A,Mvar)


    def testMul(self):
        assert self.A**2==self.A*self.A

    def testDiv(self):
        assert self.A/self.B == self.A*self.B**(-1)

    def testMvarMul(self):
        assert (self.A*self.B).cov == (self.A*self.B.transform()+self.B*self.A.transform()).cov/2
        if not self.flat: 
            assert (self.A*self.B).mean == (self.A*self.B.transform()+self.B*self.A.transform()).mean/2

        assert self.M*self.B==self.M*self.B.transform()
        if not self.flat:
            assert self.A**2==self.A*self.A.transform()

        assert self.A*(self.B**0+self.B**0) == self.A*(2*self.B**0)
        assert (self.A*self.B**0 + self.A*self.B**0).cov == (2*self.A*self.B**0).cov 

        assert self.A*self.A==self.A**2
        assert self.A*self.B == self.B*self.A
        if not self.flat:
            assert self.A*self.A==self.A*self.A.transform()


    def testScalarMul(self):
        assert self.A+self.A == 2*self.A
        assert (2*self.A).mean==2*self.A.mean
        assert (2*self.A.cov) == 2*self.A.cov
    
        assert self.A*(self.K1+self.K2)==self.A*self.K1+self.A*self.K2

        assert (self.A*(self.K1*self.E)).mean == self.K1*self.A.mean
        assert (self.A*(self.K1*self.E)).cov == self.A.cov*abs(self.K1)**2
        assert (self.K1*self.A)*self.K2 == self.K1*(self.A*self.K2)

        assert (2*self.B**0).transform() == sqrt(2)*(self.B**0).transform()    
        
        assert self.A*self.K1 == self.K1*self.A

        assert (self.A*self.K1).mean==self.K1*self.A.mean
        assert (self.A*self.K1).cov== (self.A.cov)*self.K1

        assert (self.A*(self.E*self.K1)).mean==self.A.mean*self.K1
        assert (self.A*(self.E*self.K1)).cov ==(self.E*self.K1).H*self.A.cov*(self.E*self.K1)


    def testMixedMul(self):
        assert self.K1*self.A*self.M == self.A*self.K1*self.M 
        assert self.K1*self.A*self.M == self.A*self.M*self.K1

    def testMatrixMul(self):
        assert (self.A*self.M)*self.M2 == self.A*(self.M*self.M2)
        assert (self.M*self.M2)*self.A == self.M*(self.M2*self.A)

        assert (self.A*self.M).cov == self.M.H*self.A.cov*self.M
        assert (self.A**2).transform() == self.A.cov
        
        assert self.A*self.E==self.A
        assert (-self.A)*self.E==-self.A 

        assert (self.A*self.M).cov==self.M.H*self.A.cov*self.M
        assert (self.A*self.M).mean==self.A.mean*self.M

    def testDiv(self):
        assert self.A/self.B == self.A*(self.B**(-1))
        assert self.A/self.M == self.A*(self.M**(-1))
        assert self.A/self.K1 == self.A*(self.K1**(-1))
        assert self.K1/self.A == self.K1*(self.A**(-1))
        assert self.M/self.A==self.M*(self.A**(-1))


class propertyTester(myTests):
    def testNdim(self):
        assert self.A.ndim == self.A.mean.size

    def testShape(self):
        assert self.A.vectors.shape == self.A.shape
        assert (self.A.var.size,self.A.mean.size)==self.A.shape
        assert self.A.shape[1]==self.A.ndim

    def testCov(self):
        assert self.A.vectors.H*numpy.diagflat(self.A.var)*self.A.vectors == self.A.cov
        assert self.A.transform()**2 == abs(self.A).cov 
        assert self.A.transform(2) == abs(self.A).cov
        if not(self.flat and self.N<0):
            assert self.A.transform()**self.N == self.A.transform(self.N)

    def testScaled(self):
        assert self.A.scaled.H*self.A.scaled==abs(self.A).cov
        assert Matrix(helpers.mag2(self.A.scaled))==self.A.var
        assert self.A.vectors.H*self.A.scaled==self.A.transform()

    def testVectors(self):
        if self.A.shape[0] == self.A.shape[1]:
            assert self.A.vectors.H*self.A.vectors==Matrix.eye
        else:
            a=self.A.inflate()
            assert a.vectors.H*a.vectors==Matrix.eye

    def testTransform(self):
        assert self.A.transform() == self.A.transform(1)
        assert self.A.transform() == self.A.scaled.H*self.A.vectors
    
    def testCov(self):
        assert self.A.cov == (self.A**2).transform()
        assert self.A.cov == self.A.transform()*self.A.transform()
        assert self.A.cov == self.A.transform()**2
        assert self.A.cov == self.A.transform(2)
        assert (self.A*self.B.transform() + self.B*self.A.transform()).cov/2 == (self.A*self.B).cov


class mergeTester(myTests):
    def testStack(self):
        self.AB= Mvar.stack(self.A,self.B)
        assert self.AB[:self.A.ndim]==self.A
        assert self.AB[self.A.ndim:]==self.B
        assert Mvar.stack(Mvar.infs(2),Mvar.infs(5))==Mvar.infs(7)
        assert Mvar.stack(Mvar.zeros(2),Mvar.zeros(5))==Mvar.zeros(7)
        

class powerTester(myTests):
    def testIntPowers(self):
        assert self.A.transform(self.N)== (self.A**self.N).transform()
        assert self.A.transform(self.N)== self.A.transform()**self.N 

    def testRealPowers(self):
        assert (self.A**self.K1.real).transform() == self.A.transform(self.K1.real) 

    def testComplexPowers(self):
        assert (self.A**self.K1).transform() == self.A.transform(self.K1) 

    def testZeroPow(self):
        assert self.A**0*self.A==self.A
        assert self.A*self.A**0==self.A
        assert Matrix((self.A**0).var) == numpy.ones

        if not self.flat:
            assert self.A**0 == self.A**(-1)*self.A
            assert self.A**0 == self.A*self.A**(-1)
            assert self.A**0 == self.A/self.A 
            assert (self.A**0).mean == self.A.mean*(self.A**-1).transform()
            assert (self.A**0).mean == self.A.mean*self.A.transform(-1)


    def testOnePow(self):
        assert self.A==self.A**1
        assert -self.A == (-self.A)**1

        if not self.flat:
            assert self.A == (self.A**-1)**-1

    def testRealPow(self):
        assert self.A*self.A==self.A**2
        assert self.A/self.A**-1 == self.A**2
        
        if not self.flat:
            k=self.K1.real
            assert self.A**k == self.A*self.A.transform(k-1) + Mvar(mean=self.A.mean-self.A.mean*self.A.transform(0)) 
    
        assert self.A.mean*self.A.transform(0) == ((self.A**-1)**-1).mean

    
        assert (self.A**self.K1.real)*(self.A**self.K2.real)==self.A**(self.K1.real+self.K2.real)
        assert self.A**self.K1.real/self.A**self.K2.real==self.A**(self.K1-self.K2)

class linalgTester(myTests):
    def testTrace(self):
        assert Matrix(numpy.trace(self.A.transform(0))) == self.A.shape[0] 
        assert Matrix(self.A.trace()) == self.A.var.sum()
        assert Matrix(self.A.trace()) == numpy.trace(self.A.cov)

    def testDet(self):
        assert Matrix(self.A.det())== numpy.linalg.det(self.A.cov)
        assert Matrix(self.A.det()) == (
             0 if 
             self.A.shape[0]!=self.A.shape[1] else 
             numpy.prod(self.A.var)
        )

    def testDist2(self):
        if not self.flat:
            assert Matrix((self.A**0).dist2(numpy.zeros((1,self.ndim))))==helpers.mag2((self.A**0).mean)

    def testSquare(myTest):
        vectors=Matrix(helpers.ascomplex(numpy.random.randn(
            numpy.random.randint(1,10),numpy.random.randint(1,10),2
        )))
        cov = vectors.H*vectors
        Xcov = vectors*vectors.H 
        (Xval,Xvec) = numpy.linalg.eigh(Xcov)
        vec = Xvec.H*vectors
        assert vec.H*vec == cov

class givenTester(myTests):            
    def testGivenScalar(self):
        a = self.A.given(index=0,value=1)
        assert a.mean[:,0]==1
        assert a.vectors[:,0]==numpy.zeros

        a=self.A.copy(deep=True)
        a[0]=1
        assert a==self.A.given(index=0,value=1)


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

    def testMooreGiven(self):
        assert mooreGiven(self.A,index=0,value=1)==self.A.given(index=0,value=1)[1:]


class inversionTester(myTests):
    def testAbs(self):
        assert (self.A.var>=0).all()
        assert abs(self.A) == abs(~self.A)

    def testNeg(self):
        self.IA=self.A
        self.IA.var= -self.IA.var
        assert self.IA == ~self.A

        assert Matrix((~self.A).var) == (-self.A).var 
        assert Matrix((~self.A).var) == -(self.A.var)

    def testInvariant(self):
        assert (~self.A).mean == self.A.mean
        assert (~self.A).vectors==self.A.vectors
        assert (~self.A).cov == (-self.A).cov 
        assert (~self.A).cov == -(self.A.cov)
        
    def testDoubleNegative(self):
        assert ~~self.A==self.A
        assert ~(~self.A&~self.B) == self.A & self.B 
        assert (~self.A & ~self.B) == ~(self.A & self.B)


    def testParadoxes(self):
        assert (self.A & ~self.A) == Mvar(mean=self.A.mean, vectors=self.A.vectors, var=Matrix.infs)
        assert  self.A &(self.B & ~self.B) == self.A & Mvar(mean=self.B.mean, vectors=self.B.vectors, var=Matrix.infs)
        assert (self.A&~self.B) & self.B == (self.A&self.B) & ~self.B

        assert (self.A & ~self.A) == Mvar(mean=numpy.zeros(self.A.ndim))**-1
        assert self.A == self.A & (self.B & ~self.B)
        assert (self.A&self.B) & ~self.B == self.A & (self.B&~self.B)

        assert not numpy.isfinite((self.A & ~self.A).var).any()

        P=self.A.copy()
        P.var=P.var/0.0
        assert P==(self.A & ~self.A)       


class blendTester(myTests):
    def testCommutativity(self):
        assert self.A & self.B == self.B & self.A 
        
    def testSelf(self):
        assert (self.A & self.A).cov == self.A.cov/2
        assert (self.A & self.A).mean == self.A.mean
        
    def testNotFlat(self):
        if not self.flat:
            assert self.A & self.B == 1/(1/self.A+1/self.B)
            assert self.A & -self.A == Mvar(mean=numpy.zeros(self.ndim))**-1
            assert self.A & ~self.A == Mvar(mean=numpy.zeros(self.ndim))**-1
            assert self.A & self.B == wiki(self.A,self.B)
            
            abc=numpy.random.permutation([self.A,self.B,self.C])
            assert self.A & self.B & self.C == helpers.paralell(*abc)
            assert self.A & self.B & self.C == reduce(operator.and_ ,abc)
    
            assert (self.A & self.B) & self.C == self.A & (self.B & self.C)
   
            assert self.A**-1 == self.A*self.A**-2
            assert self.A & self.B == (self.A*self.A**-2+self.B*self.B**-2)**-1

            D = self.A*(self.A.cov)**(-1) + self.B*(self.B.cov)**(-1)
            assert wiki(self.A,self.B) == D*(D.cov)**(-1)
            assert self.A & self.B == wiki(self.A,self.B)


    def testKnownValues(self):
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


if __name__ == '__main__':
    unittest.main()
