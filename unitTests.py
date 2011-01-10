#! /usr/bin/env python

import unittest
import operator
import cPickle
import copy
import sys

import numpy
import itertools

import mvar
from mvar import sqrt
from mvar import Mvar
from matrix import Matrix
import helpers

import testObjects


class myTests(unittest.TestCase):
    def setUp(self):
       self.__dict__.update(cPickle.loads(self.jar))


class commuteTester(myTests):
   def testRight(self):
        self.assertTrue( self.A+self.B == self.B+self.A )
        self.assertTrue( self.A*self.B == self.B*self.A )
        self.assertTrue( self.B-self.A == (-self.A)+self.B )
        self.assertTrue( self.A/self.B == (self.B**-1)*self.A )
        self.assertTrue( self.A & self.B == self.B & self.A )
        self.assertTrue( self.A | self.B == self.B | self.A )
        self.assertTrue( self.A ^ self.B == self.B ^ self.A )

class creationTester(myTests):
    def testFromData(self):
        self.assertTrue( Mvar.fromData(self.A)==self.A )

        data=[1,2,3]
        new=Mvar.fromData(data)
        self.assertTrue( new.mean == data )
        self.assertTrue( (new.var == numpy.zeros([0])).all() )
        self.assertTrue( new.vectors == numpy.zeros )
        self.assertTrue( new.cov == numpy.zeros([3,3]) )

    def testZeros(self):
        n=max(abs(self.N),1)
        Z=Mvar.zeros(n)
        self.assertTrue( Z.mean==Matrix.zeros )
        self.assertTrue( Z.var.size==0 )
        self.assertTrue( Z.vectors.size==0 )

    def testInfs(self):
        n=abs(self.N)
        inf=Mvar.infs(n)
        self.assertTrue( inf.mean==Matrix.zeros )
        self.assertTrue( inf.var.size==inf.mean.size==n )
        self.assertTrue( (inf.var==numpy.inf).all() )
        self.assertTrue( inf.vectors==Matrix.eye )

    def testCopy(self):
        self.A2=self.A.copy(deep=True)        
        self.assertTrue( self.A2 == self.A )
        self.assertTrue( self.A2 is not self.A )
        self.assertTrue( self.A2.mean is not self.A.mean )

        self.A.copy(self.B,deep=True)
        self.assertTrue( self.B == self.A )
        self.assertTrue( self.B is not self.A )
        self.assertTrue( self.A.mean is not self.B.mean )

        self.A2=self.A.copy(deep=False)        
        self.assertTrue( self.A2 == self.A )
        self.assertTrue( self.A2 is not self.A )
        self.assertTrue( self.A2.mean is self.A.mean )

        self.A.copy(self.B,deep=False)
        self.assertTrue( self.B == self.A )
        self.assertTrue( self.B is not self.A )
        self.assertTrue( self.A.mean is self.B.mean )


class equalityTester(myTests):
    def testEq(self):
        # always equal if same object
        self.assertTrue( self.A==self.A )
        # still equal if copy
        self.assertTrue( self.A==self.A.copy() )
        self.assertTrue( self.A is not self.A.copy() )
    
    def testCosmetic(self):
        self.assertTrue( self.A==self.A.square() )
        self.assertTrue( self.A==self.A.copy() )
        self.assertTrue( self.A==self.A.inflate() )
        self.assertTrue( self.A==self.A.inflate().squeeze() )

    def testInf(self):
        self.assertTrue(
             Mvar(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
             Mvar(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
        )
        
    def testNot(self):
        self.assertTrue( self.A!=self.B )

class signTester(myTests):
    def testPlus(self):
        self.assertTrue( self.A == +self.A )
        self.assertTrue( self.A == ++self.A )
        self.assertTrue( self.A is not +self.A )
        self.assertTrue( self.A+self.A == 2*self.A )
        self.assertTrue( self.A+self.A+self.A == 3*self.A )

    def testMinus(self):
        self.assertTrue( -self.A == -1*self.A )
        self.assertTrue( self.A == --self.A )
        self.assertTrue( self.A--self.B==self.A+self.B )
        self.assertTrue( self.A-self.B == self.A+(-self.B) )

    def testPos(self):
        self.assertTrue( self.A==+self.A==++self.A )
    
    def testNeg(self):
        self.assertTrue( -self.A==-1*self.A )
        self.assertTrue( self.A==-1*-1*self.A )
        self.assertTrue( 1j*self.A*1j==1j*(1j*self.A) ==-self.A )

    def testsAdd(self):
        self.assertTrue( (self.A+self.A)==(2*self.A) )

        self.assertTrue( (self.A+self.A).mean==(2*self.A).mean )
        self.assertTrue( (self.A+self.A).mean==2*self.A.mean )

        n=abs(self.N)
        self.assertTrue( numpy.array(list(itertools.repeat(self.A,n))).sum() == self.A*n )

        self.assertTrue( self.A+self.B ==Mvar(
            mean=self.A.mean+self.B.mean,
            vectors=numpy.vstack([self.A.vectors,self.B.vectors]),
            var = numpy.concatenate([self.A.var,self.B.var]),
        ))

        self.assertTrue( (self.A+self.A).mean==(2*self.A).mean )
        self.assertTrue( (self.A+self.A).mean==2*self.A.mean )

        self.assertTrue( (self.A+self.B).mean==self.A.mean+self.B.mean )
        self.assertTrue( (self.A+self.B).cov==self.A.cov+self.B.cov )

    def testSub(self):
        self.assertTrue( self.B+(-self.A) == self.B+(-1)*self.A == self.B-self.A )
        self.assertTrue( (self.B-self.A)+self.A==self.B )

        self.assertTrue( self.A-self.A == Mvar(mean=numpy.zeros_like(self.A.mean)) )
        self.assertTrue( (self.A-self.B)+self.B == self.A )
        self.assertTrue( (self.A-self.B).mean == self.A.mean - self.B.mean )
        self.assertTrue( (self.A-self.B).cov== self.A.cov - self.B.cov )

        self.assertTrue( (self.A+self.B*(-self.E)).mean == self.A.mean - self.B.mean )
        self.assertTrue( (self.A+self.B*(-self.E)).cov== self.A.cov + self.B.cov )

        self.assertTrue( self.A-self.B == -(self.B-self.A) )
        self.assertTrue( self.A+(self.B-self.B)==self.A )

class productTester(myTests):
    def testMulTypes(self):
        self.assertTrue( isinstance(self.A*self.B,Mvar) )
        self.assertTrue( isinstance(self.A*self.M,Mvar) )
        self.assertTrue( isinstance(self.M*self.A,Matrix) )
        self.assertTrue( isinstance(self.A*self.K1,Mvar) )
        self.assertTrue( isinstance(self.K1*self.A,Mvar) )


    def testMul(self):
        self.assertTrue( self.A**2==self.A*self.A )

    def testDiv(self):
        self.assertTrue( self.A/self.B == self.A*self.B**(-1) )

    def testMvarMul(self):
        self.assertTrue( (self.A*self.B).cov == (self.A*self.B.transform()+self.B*self.A.transform()).cov/2 )
        if not (self.A.flat or self.B.flat): 
            self.assertTrue( (self.A*self.B).mean == (self.A*self.B.transform()+self.B*self.A.transform()).mean/2 )

        self.assertTrue( self.M*self.B==self.M*self.B.transform() )
        if not self.A.flat:
            self.assertTrue( self.A**2==self.A*self.A.transform() )

        self.assertTrue( self.A*(self.B**0+self.B**0) == self.A*(2*self.B**0) )
        self.assertTrue( (self.A*self.B**0 + self.A*self.B**0).cov == (2*self.A*self.B**0).cov )

        self.assertTrue( self.A*self.A==self.A**2 )
        self.assertTrue( self.A*self.B == self.B*self.A )
        if not self.A.flat:
            self.assertTrue( self.A*self.A==self.A*self.A.transform() )


    def testScalarMul(self):
        self.assertTrue( self.A+self.A == 2*self.A )
        self.assertTrue( (2*self.A).mean==2*self.A.mean )
        self.assertTrue( (2*self.A.cov) == 2*self.A.cov )
    
        self.assertTrue( self.A*(self.K1+self.K2)==self.A*self.K1+self.A*self.K2 )

        self.assertTrue( (self.A*(self.K1*self.E)).mean == self.K1*self.A.mean )
        self.assertTrue( (self.A*(self.K1*self.E)).cov == self.A.cov*abs(self.K1)**2 )
        self.assertTrue( (self.K1*self.A)*self.K2 == self.K1*(self.A*self.K2) )

        self.assertTrue( (2*self.B**0).transform() == sqrt(2)*(self.B**0).transform() )
        
        self.assertTrue( self.A*self.K1 == self.K1*self.A )

        self.assertTrue( (self.A*self.K1).mean==self.K1*self.A.mean )
        self.assertTrue( (self.A*self.K1).cov== (self.A.cov)*self.K1 )

        self.assertTrue( (self.A*(self.E*self.K1)).mean==self.A.mean*self.K1 )
        self.assertTrue( (self.A*(self.E*self.K1)).cov ==(self.E*self.K1).H*self.A.cov*(self.E*self.K1) )


    def testMixedMul(self):
        self.assertTrue( self.K1*self.A*self.M == self.A*self.K1*self.M )
        self.assertTrue( self.K1*self.A*self.M == self.A*self.M*self.K1 )

    def testMatrixMul(self):
        self.assertTrue( (self.A*self.M)*self.M2 == self.A*(self.M*self.M2) )
        self.assertTrue( (self.M*self.M2)*self.A == self.M*(self.M2*self.A) )

        self.assertTrue( (self.A*self.M).cov == self.M.H*self.A.cov*self.M )
        self.assertTrue( (self.A**2).transform() == self.A.cov )
        
        self.assertTrue( self.A*self.E==self.A )
        self.assertTrue( (-self.A)*self.E==-self.A )

        self.assertTrue( (self.A*self.M).cov==self.M.H*self.A.cov*self.M )
        self.assertTrue( (self.A*self.M).mean==self.A.mean*self.M )

    def testDiv(self):
        self.assertTrue( self.A/self.B == self.A*(self.B**(-1)) )
        self.assertTrue( self.A/self.M == self.A*(self.M**(-1)) )
        self.assertTrue( self.A/self.K1 == self.A*(self.K1**(-1)) )
        self.assertTrue( self.K1/self.A == self.K1*(self.A**(-1)) )
        self.assertTrue( self.M/self.A==self.M*(self.A**(-1)) )


class propertyTester(myTests):
    def testNdim(self):
        self.assertTrue( self.A.ndim == self.A.mean.size )

    def testShape(self):
        self.assertTrue( self.A.vectors.shape == self.A.shape )
        self.assertTrue( (self.A.var.size,self.A.mean.size)==self.A.shape )
        self.assertTrue( self.A.shape[1]==self.A.ndim )

    def testCov(self):
        self.assertTrue( self.A.vectors.H*numpy.diagflat(self.A.var)*self.A.vectors == self.A.cov )
        self.assertTrue( self.A.transform()**2 == abs(self.A).cov )
        self.assertTrue( self.A.transform(2) == abs(self.A).cov )
        if not(self.A.flat and self.N<0):
            self.assertTrue( self.A.transform()**self.N == self.A.transform(self.N) )

    def testScaled(self):
        self.assertTrue( self.A.scaled.H*self.A.scaled==abs(self.A).cov )
        self.assertTrue( Matrix(helpers.mag2(self.A.scaled))==self.A.var )
        self.assertTrue( self.A.vectors.H*self.A.scaled==self.A.transform() )

    def testVectors(self):
        if self.A.shape[0] == self.A.shape[1]:
            self.assertTrue( self.A.vectors.H*self.A.vectors==Matrix.eye )
        else:
            a=self.A.inflate()
            self.assertTrue( a.vectors.H*a.vectors==Matrix.eye )

    def testTransform(self):
        self.assertTrue( self.A.transform() == self.A.transform(1) )
        self.assertTrue( self.A.transform() == self.A.scaled.H*self.A.vectors )
    
    def testCov(self):
        self.assertTrue( self.A.cov == (self.A**2).transform() )
        self.assertTrue( self.A.cov == self.A.transform()*self.A.transform() )
        self.assertTrue( self.A.cov == self.A.transform()**2 )
        self.assertTrue( self.A.cov == self.A.transform(2) )
        self.assertTrue( (self.A*self.B.transform() + self.B*self.A.transform()).cov/2 == (self.A*self.B).cov )


class mergeTester(myTests):
    def testStack(self):
        self.AB= Mvar.stack(self.A,self.B)
        self.assertTrue( self.AB[:self.A.ndim]==self.A )
        self.assertTrue( self.AB[self.A.ndim:]==self.B )
        self.assertTrue( Mvar.stack(Mvar.infs(2),Mvar.infs(5))==Mvar.infs(7) )
        self.assertTrue( Mvar.stack(Mvar.zeros(2),Mvar.zeros(5))==Mvar.zeros(7) )
        

class powerTester(myTests):
    def testIntPowers(self):
        self.assertTrue( self.A.transform(self.N)== (self.A**self.N).transform() )
        self.assertTrue( self.A.transform(self.N)== self.A.transform()**self.N )
        
    def testRealPowers(self):
        k=numpy.real(self.K1)
        self.assertTrue( (self.A**k).transform() == self.A.transform(k) )

    def testComplexPowers(self):
        self.assertTrue( (self.A**self.K1).transform() == self.A.transform(self.K1) )
        self.assertTrue( self.A**self.K1*self.A**self.K2 == self.A**(self.K1+self.K2))
        self.assertTrue( self.A**self.K1/self.A**self.K2 == self.A**(self.K1-self.K2))


    def testZeroPow(self):
        self.assertTrue( self.A**0*self.A==self.A )
        self.assertTrue( self.A*self.A**0==self.A )
        self.assertTrue( Matrix((self.A**0).var) == numpy.ones )

    def testZeroFlat(self):
        if not self.A.flat:
            self.assertTrue( self.A**0 == self.A**(-1)*self.A )
            self.assertTrue( self.A**0 == self.A*self.A**(-1) )
            self.assertTrue( self.A**0 == self.A/self.A )
            self.assertTrue( (self.A**0).mean == self.A.mean*(self.A**-1).transform() )
            self.assertTrue( (self.A**0).mean == self.A.mean*self.A.transform(-1) )


    def testOnePow(self):
        self.assertTrue( self.A==self.A**1 )
        self.assertTrue( -self.A == (-self.A)**1 )

        if not self.A.flat:
            self.assertTrue( self.A == (self.A**-1)**-1 )

    def testRealPow(self):
        self.assertTrue( self.A*self.A==self.A**2 )
        self.assertTrue( self.A/self.A**-1 == self.A**2 )
        
        self.assertTrue( self.A.mean*self.A.transform(0) == ((self.A**-1)**-1).mean )

        k1=numpy.real(self.K1)
        k2=numpy.real(self.K2)
        self.assertTrue( (self.A**k1)*(self.A**k2)==self.A**(k1+k2) )
        self.assertTrue( self.A**k1/self.A**k2==self.A**(k1-k2) )

        if not self.A.flat:
            self.assertTrue( self.A**k1 == self.A*self.A.transform(k1-1) + Mvar(mean=self.A.mean-self.A.mean*self.A.transform(0)) )
        

class linalgTester(myTests):
    def testTrace(self):
        self.assertTrue( Matrix(numpy.trace(self.A.transform(0))) == self.A.shape[0] )
        self.assertTrue( Matrix(self.A.trace()) == self.A.var.sum() )
        self.assertTrue( Matrix(self.A.trace()) == numpy.trace(self.A.cov) )

    def testDet(self):
        self.assertTrue( Matrix(self.A.det())== numpy.linalg.det(self.A.cov) )
        self.assertTrue( Matrix(self.A.det()) == 
             0 if 
             self.A.shape[0]!=self.A.shape[1] else 
             numpy.prod(self.A.var)
        )

    def testDist2(self):
        if not self.A.flat:
            self.assertTrue( Matrix((self.A**0).dist2(numpy.zeros((1,self.ndim))))==helpers.mag2((self.A**0).mean) )

    def testSquare(self):
        vectors=Matrix(helpers.ascomplex(numpy.random.randn(
            numpy.random.randint(1,10),numpy.random.randint(1,10),2
        )))
        cov = vectors.H*vectors
        Xcov = vectors*vectors.H 
        (Xval,Xvec) = numpy.linalg.eigh(Xcov)
        vec = Xvec.H*vectors
        self.assertTrue( vec.H*vec == cov )

class givenTester(myTests):            
    def testGivenScalar(self):
        a = self.A.given(index=0,value=1)
        self.assertTrue( a.mean[:,0]==1 )
        self.assertTrue( a.vectors[:,0]==numpy.zeros )

        a=self.A.copy(deep=True)
        a[0]=1
        self.assertTrue( a==self.A.given(index=0,value=1) )


    def testGivenLinear(self):
        L1=Mvar(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
        L2=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf) 
        self.assertTrue( L1.given(index=0,value=1) == L1&L2 )
        self.assertTrue( (L1&L2).mean==[1,1] )
        self.assertTrue( (L1&L2).cov==[[0,0],[0,2]] )

    def testGivenMvar(self):
        Y=Mvar(mean=[0,1],vectors=Matrix.eye, var=[numpy.inf,1])
        X=Mvar(mean=[1,0],vectors=Matrix.eye,var=[1,numpy.inf])
        x=Mvar(mean=1,var=1)
        self.assertTrue( Y.given(index=0,value=x) == X&Y )

    def testMooreGiven(self):
        self.assertTrue( mvar.mooreGiven(self.A,index=0,value=1)==self.A.given(index=0,value=1)[1:] )


class inversionTester(myTests):
    def testAbs(self):
        self.assertTrue( (self.A.var>=0).all() )
        self.assertTrue( abs(self.A) == abs(~self.A) )

    def testNeg(self):
        self.IA=self.A.copy()
        self.IA.var= -self.IA.var
        self.assertTrue( self.IA == ~self.A )

        self.assertTrue( Matrix((~self.A).var) == (-self.A).var )
        self.assertTrue( Matrix((~self.A).var) == -(self.A.var) )

    def testInvariant(self):
        self.assertTrue( (~self.A).mean == self.A.mean )
        self.assertTrue( (~self.A).vectors==self.A.vectors )
        self.assertTrue( (~self.A).cov == (-self.A).cov )
        self.assertTrue( (~self.A).cov == -(self.A.cov) )
        
    def testDoubleNegative(self):
        self.assertTrue( ~~self.A==self.A )
        self.assertTrue( ~(~self.A&~self.B) == self.A & self.B )
        self.assertTrue( (~self.A & ~self.B) == ~(self.A & self.B) )


    def testParadoxes(self):
        self.assertTrue( (self.A & ~self.A) == Mvar(mean=self.A.mean, vectors=self.A.vectors, var=Matrix.infs) )
        self.assertTrue(  self.A &(self.B & ~self.B) == self.A & Mvar(mean=self.B.mean, vectors=self.B.vectors, var=Matrix.infs) )
        self.assertTrue( (self.A&~self.B) & self.B == (self.A&self.B) & ~self.B )

        self.assertTrue( (self.A & ~self.A) == Mvar(mean=numpy.zeros(self.A.ndim))**-1 )
        self.assertTrue( self.A == self.A & (self.B & ~self.B) )
        self.assertTrue( (self.A&self.B) & ~self.B == self.A & (self.B&~self.B) )

        self.assertTrue( not numpy.isfinite((self.A & ~self.A).var).any() )

        P=self.A.copy()
        P.var=P.var/0.0
        self.assertTrue( P==(self.A & ~self.A) )



class blendTester(myTests):
    def testCommutativity(self):
        self.assertTrue( self.A & self.B == self.B & self.A)
        
    def testSelf(self):
        self.assertTrue( (self.A & self.A).cov == self.A.cov/2)
        self.assertTrue( (self.A & self.A).mean == self.A.mean)
        
    def testNotFlat(self):
        if not (self.A.flat or self.B.flat):
            self.assertTrue( self.A & self.B == 1/(1/self.A+1/self.B))
            self.assertTrue( self.A & -self.A == Mvar(mean=numpy.zeros(self.ndim))**-1)
            self.assertTrue( self.A & ~self.A == Mvar(mean=numpy.zeros(self.ndim))**-1)
            self.assertTrue( self.A & self.B == mvar.wiki(self.A,self.B))
               
            self.assertTrue( self.A**-1 == self.A*self.A**-2)
            self.assertTrue( self.A & self.B == (self.A*self.A**-2+self.B*self.B**-2)**-1)

            D = self.A*(self.A.cov)**(-1) + self.B*(self.B.cov)**(-1)
            self.assertTrue( mvar.wiki(self.A,self.B) == D*(D.cov)**(-1))
            self.assertTrue( self.A & self.B == mvar.wiki(self.A,self.B))

        if not (self.A.flat or self.B.flat or self.C.flat):
            abc=numpy.random.permutation([self.A,self.B,self.C])
            self.assertTrue( self.A & self.B & self.C == helpers.paralell(*abc))
            self.assertTrue( self.A & self.B & self.C == reduce(operator.and_ ,abc))
    
            self.assertTrue( (self.A & self.B) & self.C == self.A & (self.B & self.C))



    def testKnownValues1(self):
        L1=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf)
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[1,1])
        self.assertTrue( (L1&L2).var.size==0)

    def testKnownValues2(self):
        L1=Mvar(mean=[0,0],vectors=[1,1],var=numpy.inf)
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[1,1])
        self.assertTrue( (L1&L2).var.size==0)

    def testKnownValues3(self):
        L1=Mvar(mean=[0,0],vectors=Matrix.eye, var=[1,1])
        L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[0,1])
        self.assertTrue( (L1&L2).var==1)
        self.assertTrue( (L1&L2).vectors==[1,0])

def getTests(fixture=None):
    testCases= [
            value for (name,value) in copy.copy(globals()).iteritems() 
            if isinstance(value,type) and issubclass(value,myTests)
        ]
    if fixture is None:
        return testCases
   
    jar=cPickle.dumps(fixture)
    testCases = [
        unittest.makeSuite(
            type(tc.__name__,(tc,),{'jar':jar})
        ) for tc in testCases
    ]

    return testCases

if __name__=='__main__':
    suite=unittest.TestSuite()

    if '-r' in sys.argv:
        testFixture=testObjects.testObjects
    else:
        testFixture=testObjects.makeObjects()
    
    suite.addTests(getTests(testFixture))


    unittest.TextTestRunner().run(suite)
