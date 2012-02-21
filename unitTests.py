#! /usr/bin/env python

import unittest
import operator
import cPickle
import copy
import sys

import numpy
import itertools

import mvn
sqrt = mvn.sqrt 
Mvn = mvn.Mvn
Matrix = mvn.Matrix

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
        self.assertTrue( Mvn.fromData(self.A)==self.A )

        data=[1,2,3]
        one = Mvn.fromData(data)
        self.assertTrue(one.ndim == 1)


        data=[[1,2,3]]
        many=Mvn.fromData(data)
        self.assertTrue( many.mean == data )
        self.assertTrue( Matrix(many.var) == numpy.zeros )
        self.assertTrue( many.vectors == numpy.zeros )
        self.assertTrue( many.cov == numpy.zeros )

    def testFromCov(self):
        self.assertTrue(Mvn.fromCov(self.A.cov,mean=self.A.mean) == self.A)

    def testZeros(self):
        n=abs(self.N)
        Z=Mvn.zeros(n)
        self.assertTrue( Z.mean==Matrix.zeros )
        self.assertTrue( Z.var.size==0 )
        self.assertTrue( Z.vectors.size==0 )
        self.assertTrue( Z**-1 == Mvn.infs)

    def testInfs(self):
        n=abs(self.N)
        inf=Mvn.infs(n)
        self.assertTrue( inf.mean==Matrix.zeros )
        self.assertTrue( inf.var.size==inf.mean.size==n )
        self.assertTrue( Matrix(inf.var)==Matrix.infs )
        self.assertTrue( inf.vectors==Matrix.eye )
        self.assertTrue( inf**-1 == Mvn.zeros )
    
    def testEye(self):
        n=abs(self.N)
        eye=Mvn.eye(n)
        self.assertTrue(eye.mean==Matrix.zeros)
        self.assertTrue(eye.var.size==eye.mean.size==n)
        self.assertTrue(Matrix(eye.var)==Matrix.ones)
        self.assertTrue(eye.vectors==Matrix.eye)
        self.assertTrue(eye**-1 == eye)
        
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

class densityTester(myTests):
    def testDensity(self):
        """
        Another way to think of the & operation is as doing a 
        pointwize product of the probability (densities) and then 
        normalizing the total probability of 1. 

        But it works in both directions, blend and un-blend
        is like multiply and divide.
        """
        if not (self.A.flat or self.B.flat):
            #remember you can undo a blend.
            self.assertTrue((~self.B) & self.A & self.B == self.A)

            #setup
            AB  = self.A &  self.B
            A_B = self.A & ~self.B
                
            locations = AB.sample([10,10])

            # A&B == k*A.*B
            Da = self.A.density(locations)
            Db = self.B.density(locations)

            Dab  = (AB).density(locations)

            ratio = Dab /(Da*Db)
            self.assertTrue(Matrix(0) == ratio.var())

            # A&(~B) == k*A./B
            Da_b = (A_B).density(locations)
            ratio = Da_b/(Da/Db)
            self.assertTrue(Matrix(0) == ratio.var())

            #log
            Ea = self.A.entropy(locations)
            Eb = self.B.entropy(locations)

            Eab  = (AB).entropy(locations)
            delta = Eab-(Ea+Eb)
            self.assertTrue(Matrix(0) == delta.var())

            Ea_b = (A_B).entropy(locations)
            delta = Ea_b-(Ea-Eb)
            self.assertTrue(Matrix(0) == delta.var())


    def testDensity2(self):
        data = self.A.sample([5,5])
        self.assertTrue(
            Matrix(self.A.density(data)) == 
            numpy.exp(-self.A.entropy(data))
        )

        





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
             Mvn(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
             Mvn(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
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

    def testAdd(self):
        self.assertTrue( (self.A+self.A)==(2*self.A) )

        self.assertTrue( (self.A+self.A).mean==(2*self.A).mean )
        self.assertTrue( (self.A+self.A).mean==2*self.A.mean )

        n=abs(self.N)
        self.assertTrue( sum(itertools.repeat(self.A,n),Mvn.zeros(self.A.ndim)) == self.A*n )
        self.assertTrue( sum(itertools.repeat(-self.A,n),Mvn.zeros(self.A.ndim)) == self.A*(-n) )
        

        self.assertTrue( self.A+self.B ==Mvn(
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

        self.assertTrue( self.A-self.A == Mvn(mean=numpy.zeros_like(self.A.mean)) )
        self.assertTrue( (self.A-self.B)+self.B == self.A )
        self.assertTrue( (self.A-self.B).mean == self.A.mean - self.B.mean )
        self.assertTrue( (self.A-self.B).cov== self.A.cov - self.B.cov )

        self.assertTrue( (self.A+self.B*(-self.E)).mean == self.A.mean - self.B.mean )
        self.assertTrue( (self.A+self.B*(-self.E)).cov== self.A.cov + self.B.cov )

        self.assertTrue( self.A-self.B == -(self.B-self.A) )
        self.assertTrue( self.A+(self.B-self.B)==self.A )


class productTester(myTests):
    def testMulTypes(self):
        self.assertTrue( isinstance(self.A*self.B,Mvn) )
        self.assertTrue( isinstance(self.A*self.M,Mvn) )
        self.assertTrue( isinstance(self.M.T*self.A,Matrix) )
        self.assertTrue( isinstance(self.A*self.K1,Mvn) )
        self.assertTrue( isinstance(self.K1*self.A,Mvn) )


    def testMul(self):
        self.assertTrue( self.A**2==self.A*self.A )

    def testMvnMul(self):
        self.assertTrue( (self.A*self.B).cov == (self.A*self.B.transform()+self.B*self.A.transform()).cov/2 )
        if not (self.A.flat or self.B.flat): 
            self.assertTrue( (self.A*self.B).mean == (self.A*self.B.transform()+self.B*self.A.transform()).mean/2 )

        self.assertTrue( self.M.T*self.B==self.M.T*self.B.transform() )
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
        self.assertTrue( (self.A*self.M)*self.M2.H == self.A*(self.M*self.M2.H) )
        self.assertTrue( (self.M*self.M2.H)*self.A == self.M*(self.M2.H*self.A) )

        self.assertTrue( (self.A*self.M).cov == self.M.H*self.A.cov*self.M )
        self.assertTrue( (self.A**2).transform() == self.A.cov )
        
        self.assertTrue( self.A*self.E==self.A )
        self.assertTrue( (-self.A)*self.E==-self.A )

        self.assertTrue( (self.A*self.M).cov==self.M.H*self.A.cov*self.M )
        self.assertTrue( (self.A*self.M).mean==self.A.mean*self.M )

    def testDiv(self):
        self.assertTrue( self.A/self.B == self.A*self.B**(-1) )
        
        m=self.M*self.M2.T
        self.assertTrue( self.A/m == self.A*(m**(-1)) )
        self.assertTrue( self.A/self.K1 == self.A*(self.K1**(-1)) )
        self.assertTrue( self.K1/self.A == self.K1*(self.A**(-1)) )
        self.assertTrue( self.M.H/self.A==self.M.H*(self.A**(-1)) )


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
            
    def testCov2(self):
        self.assertTrue( self.A.cov == (self.A**2).transform() )
        self.assertTrue( self.A.cov == self.A.transform()*self.A.transform() )
        self.assertTrue( self.A.cov == self.A.transform()**2 )
        self.assertTrue( self.A.cov == self.A.transform(2) )
        self.assertTrue( 
            (self.A*self.B.transform() 
            + self.B*self.A.transform()).cov/2 
            == (self.A*self.B).cov 
        )

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
        
        self.assertTrue(self.A.vectors*self.A.vectors.H == Matrix.eye)
        self.assertTrue((self.A*self.M).cov==self.M.H*self.A.cov*self.M)
        self.assertTrue((self.A*self.M).vectors*(self.A*self.M).vectors.H == Matrix.eye)


    def testTransform(self):
        self.assertTrue( self.A.transform() == self.A.transform(1) )
        self.assertTrue( self.A.transform() == self.A.scaled.H*self.A.vectors )


class mergeTester(myTests):
    def testStack(self):
        self.AB= Mvn.stack(self.A,self.B)
        self.assertTrue( self.AB[:,:self.A.ndim]==self.A )
        self.assertTrue( self.AB[:,self.A.ndim:]==self.B )
        self.assertTrue( Mvn.stack(Mvn.infs(2),Mvn.infs(5))==Mvn.infs(7) )
        self.assertTrue( Mvn.stack(Mvn.zeros(2),Mvn.zeros(5))==Mvn.zeros(7) )
        

class powerTester(myTests):
    def testIntPowers(self):
        N = abs(self.N)
        self.assertTrue( self.A.transform(N)== (self.A**N).transform() )
        self.assertTrue( self.A.transform(N)== self.A.transform()**N )

        N = -abs(self.N)
        if not self.A.flat:
            self.assertTrue( self.A.transform(N)== (self.A**N).transform() )
            self.assertTrue( self.A.transform(N)== self.A.transform()**N )


    def testMorePowers(self):
        self.assertTrue( (self.A**self.K1).transform()**2 == self.A.transform(self.K1)**2 )

        self.assertTrue( self.A**self.K1*self.A**self.K2 == self.A**(self.K1+self.K2))
        self.assertTrue( self.A**self.K1/self.A**self.K2 == self.A**(self.K1-self.K2))

        self.assertTrue( self.A*self.A**self.K2 == self.A**(1+self.K2))
        self.assertTrue( self.A/self.A**self.K2 == self.A**(1-self.K2))

        self.assertTrue( self.A**self.K1*self.A == self.A**(self.K1+1))
        self.assertTrue( self.A**self.K1/self.A == self.A**(self.K1-1))
 

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

        k1=self.K1
        k2=self.K2
        self.assertTrue( (self.A**k1).transform() == self.A.transform(k1) )

        self.assertTrue( (self.A**k1)*(self.A**k2)==self.A**(k1+k2) )
        self.assertTrue( self.A**k1/self.A**k2==self.A**(k1-k2) )

        if not self.A.flat:
            self.assertTrue( 
                self.A**k1 == (
                    self.A*self.A.transform(k1-1) + 
                    Mvn(mean=self.A.mean-self.A.mean*self.A.transform(0))
                ))
                
class widthTester(myTests):
    def testWidth(self):
        self.assertTrue(
            Matrix([self.A[:,n].var[0] for n in range(self.A.ndim)]) == 
            self.A.width()**2
        )

        self.assertTrue(
            Matrix(self.A.corr.diagonal()) == 
            Matrix.ones
        )

        norm = self.A/self.A.width()
        self.assertTrue(norm.corr == norm.cov)
        self.assertTrue(
            Matrix([norm[:,n].var[0] for n in range(norm.ndim)]) == 
            Matrix.ones
        )
        
        self.assertTrue(
            Matrix((self.A**0).var) ==
            Matrix.ones
        )
        
        data = self.A.sample(100)
        a = Mvn.fromData(data)
        self.assertTrue(Matrix(numpy.std (data,0)) == a.width()   )    
        self.assertTrue(Matrix(numpy.var (data,0)) == a.width()**2)
        self.assertTrue(Matrix(numpy.mean(data,0)) == a.mean      )

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
        a = self.A.given(dims=0,value=1)
        self.assertTrue( a.mean[:,0]==1 )
        self.assertTrue( a.vectors[:,0]==numpy.zeros )

        a=self.A.copy(deep=True)
        a[:,0]=1
        self.assertTrue( a==self.A.given(dims=0,value=1) )


    def testGivenLinear(self):
        L1=Mvn(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
        L2=Mvn(mean=[1,0],vectors=[0,1],var=numpy.inf) 
        self.assertTrue( L1.given(dims=0,value=1) == L1&L2 )
        self.assertTrue( (L1&L2).mean==[1,1] )
        self.assertTrue( (L1&L2).cov==[[0,0],[0,2]] )

    def testGivenMvn(self):
        Y=Mvn(mean=[0,1],vectors=Matrix.eye, var=[numpy.inf,1])
        X=Mvn(mean=[1,0],vectors=Matrix.eye,var=[1,numpy.inf])
        x=Mvn(mean=1,var=1)
        self.assertTrue( Y.given(dims=0,value=x) == X&Y )

    def testGivenVector(self):
        self.assertTrue( mvn.givenVector(self.A,dims=0,value=1)==self.A.given(dims=0,value=1) )

class chainTester(myTests):
    def testBasic(self):
        self.assertTrue( self.A.chain()==self.A*numpy.hstack([self.E,self.E]) ) 
        self.assertTrue( 
            self.A.chain(transform=self.M) ==
            self.A*numpy.hstack([self.E,self.M])
        )

    def testMoore(self):
        self.assertTrue( self.A.chain(self.B) == mvn.mooreChain(self.A,self.B) )

        b=self.B*self.M
        self.assertTrue( self.A.chain(b,self.M) == mvn.mooreChain(self.A,b,self.M) )

    def testStacks(self):
        dataA=self.A.sample(100)

        a=Mvn.fromData(dataA)

        #a and a are correlated
        self.assertTrue(
            a.chain()==
            Mvn.fromData(numpy.hstack([dataA,dataA]))
        )        
        #a and a*M are corelated        
        self.assertTrue(
            a.chain(transform=self.M) == 
            dataA*numpy.hstack([self.E,self.M])
        )

        self.assertTrue( 
            a.chain(transform=self.M) == 
            Mvn.fromData(numpy.hstack([dataA,dataA*self.M]))
        )

        self.assertTrue(
            a.chain(self.B*self.M,self.M) == 
            a.chain(transform=self.M)+Mvn.stack(Mvn.zeros(a.ndim),self.B*self.M)
        )

        

    def testAnd(self):
        """
        __and__ is a shortcut across mvn.chain and mvn.given
        this is to show the relationship

        I haven't figured yet out how a the 'transform' parameter to chain works 
        with __and__, it probably involves the psudo-inverse of the transform. 

        I think the answer is on the wikipedia kalman-filtering page
        """

        measurment = self.B.mean
        sensor=self.B.copy()
        sensor.mean-=sensor.mean

        joint = self.A.chain(sensor)
        measured = joint.copy()
        measured[:,self.ndim:]=measurment

        self.assertTrue(measured[:,:self.ndim] == self.A&self.B)


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
        self.assertTrue( (self.A & ~self.A) == Mvn(mean=self.A.mean, vectors=self.A.vectors, var=Matrix.infs) )
        self.assertTrue( (self.A & ~self.A)*self.A.vectors.H == Mvn.infs )

        self.assertTrue(  
            self.A & (self.B & ~self.B) == 
            self.A & Mvn(
                mean=self.B.mean, 
                vectors=self.B.vectors, 
                var=Matrix.infs
            )
        )

        if not self.B.flat:        
            self.assertTrue( self.A == self.A & (self.B & ~self.B) )
        
        self.assertTrue( (self.A&~self.B) & self.B == (self.A&self.B) & ~self.B )

        self.assertTrue( (self.A&self.B) & ~self.B == self.A & (self.B&~self.B) )

        self.assertTrue( not numpy.isfinite((self.A & ~self.A).var).any() )

        P=self.A.copy()
        P.var=P.var/0.0
        self.assertTrue( P==(self.A & ~self.A) )


    def testPow(self):
        self.assertTrue(
           ( self.A)**(-1) + (~self.A)**(-1) == 
           Mvn.zeros
        )
        
        self.assertTrue(
           (( self.A)**(-1) + (~self.A)**(-1))**-1 == 
           Mvn.zeros(self.A.ndim)**-1
        )    

class blendTester(myTests):
    def testCommutativity(self):
        self.assertTrue( self.A & self.B == self.B & self.A)
        
    def testSelf(self):
        self.assertTrue( (self.A & self.A).cov == self.A.cov/2)
        self.assertTrue( (self.A & self.A).mean == self.A.mean)
        
        
    def testNotFlat(self):
        if not (self.A.flat or self.B.flat):
            self.assertTrue( self.A & self.B == 1/(1/self.A+1/self.B))
            self.assertTrue( self.A & -self.A == Mvn(mean=numpy.zeros(self.ndim))**-1)
            self.assertTrue( self.A & ~self.A == Mvn(mean=numpy.zeros(self.ndim))**-1)
            self.assertTrue( self.A & self.B == mvn.wiki(self.A,self.B))
               
            self.assertTrue( self.A**-1 == self.A*self.A**-2)
            self.assertTrue( self.A & self.B == (self.A*self.A**-2+self.B*self.B**-2)**-1)

            D = self.A*(self.A.cov)**(-1) + self.B*(self.B.cov)**(-1)
            self.assertTrue( mvn.wiki(self.A,self.B) == D*(D.cov)**(-1))
            self.assertTrue( self.A & self.B == mvn.wiki(self.A,self.B))

        if not (self.A.flat or self.B.flat or self.C.flat):
            abc=numpy.random.permutation([self.A,self.B,self.C])
            self.assertTrue( self.A & self.B & self.C == helpers.paralell(*abc))
            self.assertTrue( self.A & self.B & self.C == reduce(operator.and_ ,abc))
    
            self.assertTrue( (self.A & self.B) & self.C == self.A & (self.B & self.C))


    def testKnownValues1(self):
        L1=Mvn(mean=[1,0],vectors=[0,1],var=numpy.inf)
        L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[1,1])
        self.assertTrue( (L1&L2).var.size==0)

    def testKnownValues2(self):
        L1=Mvn(mean=[0,0],vectors=[1,1],var=numpy.inf)
        L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[1,1])
        self.assertTrue( (L1&L2).var.size==0)

    def testKnownValues3(self):
        L1=Mvn(mean=[0,0],vectors=Matrix.eye, var=[1,1])
        L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        self.assertTrue( (L1&L2).mean==[0,1])
        self.assertTrue( (L1&L2).var==1)
        self.assertTrue( (L1&L2).vectors==[1,0])

class quadTester(myTests):
    def testDerivation(self):
        Na = 25

        #get some data from A
        Da=Matrix(self.A.sample(Na))

        #and remake the multivariates
        A=Mvn.fromData(Da)

        # take all the dot products
        dots=(numpy.array(Da)**2).sum(1)
        self.assertTrue( Matrix(dots) == numpy.diag(Da*Da.H) )

        Mean = Matrix(dots.mean())
        Var = Matrix(dots.var())


        self.assertTrue( Mean == numpy.trace(Da*Da.H)/Na )
        self.assertTrue( Mean == numpy.trace(Da.H*Da/Na) )
        self.assertTrue( Mean == (Da*Da.H).diagonal().mean() )

        self.assertTrue( A.cov+A.mean.H*A.mean == (Da.H*Da)/Na )

        self.assertTrue( Mean == numpy.trace(A.mean.H*A.mean + A.cov) )
        self.assertTrue( Mean == numpy.trace(A.mean.H*A.mean)+numpy.trace(A.cov) )
        self.assertTrue( Mean == A.mean*A.mean.H + A.trace() )

        #definition of variance
        self.assertTrue( Var == (numpy.array(Mean -dots)**2).mean() )

        #expand it
        self.assertTrue( 
            Var == (
                Mean**2 
                -2*numpy.multiply(Mean,dots) + dots**2 
            ).mean() 
        )

        #distribute the calls to mean()
        self.assertTrue( Var == Mean**2 - 2*Mean*dots.mean() + (dots**2).mean() )

        #but Mean == dot.mean(), so
        self.assertTrue( Var == (dots**2).mean() - Mean**2 )

        self.assertTrue( Var == (dots**2).sum()/Na - Mean**2 )

        self.assertTrue( Var == ((Da*Da.H).diagonal()**2).sum()/Na - Mean**2 )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())
            *Matrix((Da*Da.H).diagonal()).H/Na 
            -Mean**2
        )

        self.assertTrue( 
            Mean ==
            (Matrix((Da*Da.H).diagonal())
            *Matrix.ones((Na,1))/Na)
        )

        self.assertTrue( 
            Mean**2 == 
            (Matrix((Da*Da.H).diagonal())
            *Matrix.ones((Na,1))/Na)**2
        )

        self.assertTrue( 
            Mean**2 == 
            Matrix((Da*Da.H).diagonal()
            *Matrix.ones((Na,1))/Na) 
            *Matrix((Da*Da.H).diagonal()
            *Matrix.ones((Na,1))/Na)
        )

        self.assertTrue( 
            Mean**2 == 
            Matrix((Da*Da.H).diagonal())
            *Matrix.ones((Na,1))*Matrix.ones((1,Na))/Na**2 
            *Matrix((Da*Da.H).diagonal()).H
        )

        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())
            *Matrix((Da*Da.H).diagonal()).H/Na 
            -
            Matrix((Da*Da.H).diagonal())
            *Matrix.ones((Na,1))*Matrix.ones((1,Na))/Na**2 
            *Matrix((Da*Da.H).diagonal()).H
        )
            
        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())
            *Matrix((Da*Da.H).diagonal()).H/Na 
            -
            (Matrix((Da*Da.H).diagonal())
            *Matrix((Da*Da.H).diagonal()).H.sum()).sum()/Na/Na
        )

        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())/Na
            *Matrix((Da*Da.H).diagonal()).H
            -
            Matrix((Da*Da.H).diagonal())/Na
            *(numpy.trace(Da*Da.H)
            *Matrix.ones((Na,1)))/Na
        )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())/Na 
            * (
                Matrix((Da*Da.H).diagonal()).H
                -
                (numpy.trace(Da*Da.H)
                *Matrix.ones((Na,1)))/Na
            )
        )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())/Na 
            *(
                Matrix((Da*Da.H).diagonal()).H
                -Mean
            )
        )

        #there's a connection in between here that I don't understand   

        #wiki: this is the Reference value
        wVar=2*numpy.trace(A.cov*A.cov)+4*A.mean*A.cov*A.mean.H

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) 
            + 
            4*numpy.trace(
                A.mean.H*A.mean
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
            )
        )

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) 
            + 
            numpy.trace(
                4*A.mean
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
                *A.mean.H
            )
        )

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) + 
            numpy.trace(
                4*A.mean
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
                *A.mean.H
            )
        )

        self.assertTrue( 
            wVar == 
            2*numpy.trace(A.cov*A.cov)
            +
            4*A.mean*A.cov*A.mean.H
        )

        self.assertTrue(
            wVar == 
            2*(A*A).trace()
            +
            4*(A*A.mean.H).trace()
        )

        self.assertTrue(
            A.quad()==
            Mvn(
                mean= A.mean*A.mean.H + A.trace(),
                var=2*(A*A).trace()+4*(A*A.mean.H).trace()
            )
        )

class innerTester(myTests):
    def testDerivation(self):
        A=self.A
        B=self.B

        Na = 20
        Nb = 10

        N=Na*Nb

        #get some data from A and B
        Da=Matrix(A.sample(Na))
        Db=Matrix(B.sample(Nb))

        #and remake the multivariates based on the samples you just took
        A=Mvn.fromData(Da)
        B=Mvn.fromData(Db)

        # take every possible combination of dot products
        dot=numpy.array(Da*Db.H)

        #the population mean
        Mean = Matrix(dot.mean())
        #the population variance
        Var = Matrix(dot.var())

        #should equal the distribution mean
        self.assertTrue( Mean == A.mean*B.mean.H )

        #definition of variance
        self.assertTrue( Var == (numpy.array(Mean -dot)**2).mean() )

        #expand it
        self.assertTrue( 
            Var == (
                Mean**2 
                - 
                2*numpy.multiply(Mean,dot) + dot**2 
            ).mean() 
        )

        #diftribute the calls to mean()
        self.assertTrue( 
            Var == 
            Mean**2 
            -2*Mean*dot.mean() 
            +(dot**2).mean() 
        )

        #but Mean == dot.mean(), so
        self.assertTrue( 
            Var == 
            (dot**2).mean() - Mean**2
        )

        dot = Matrix(dot)

        self.assertTrue( Var == numpy.trace(dot*dot.H)/N - Mean**2 )
        
        #factor everything
        self.assertTrue( 
            Var == 
            numpy.trace(Da*Db.H*Db*Da.H)/Na/Nb 
            - 
            (A.mean*B.mean.H)**2 
        )


        #rotate the trace
        self.assertTrue( 
            Var == 
            numpy.trace(Da.H*Da*Db.H*Db)/Na/Nb 
            - 
            (A.mean*B.mean.H)**2 
        )

        #group the data's
        self.assertTrue( 
            Var == 
            numpy.trace((Da.H*Da)*(Db.H*Db))/Na/Nb 
            -
            (A.mean*B.mean.H)**2 
        )

        #distribute the N's
        self.assertTrue( 
            Var == 
            numpy.trace((Da.H*Da)/Na*(Db.H*Db)/Nb) 
            - 
            (A.mean*B.mean.H)**2 
        )

        #from the definition of mean and cov
        self.assertTrue( A.cov+A.mean.H*A.mean == (Da.H*Da)/Na )
        self.assertTrue( B.cov+B.mean.H*B.mean == (Db.H*Db)/Nb )

        #replace
        self.assertTrue( 
            Var == 
            numpy.trace(
                (A.cov+A.mean.H*A.mean)
                *(B.cov+B.mean.H*B.mean)
            )
            -
            (A.mean*B.mean.H)**2 
        )


        #multiply it out
        self.assertTrue( Var == 
            numpy.trace(
                A.cov*B.cov + 
                A.mean.H*A.mean*B.cov + 
                A.cov*B.mean.H*B.mean + 
                A.mean.H*A.mean*B.mean.H*B.mean
            ) - (A.mean*B.mean.H)**2 )

        #distribute the calls to trace
        self.assertTrue( Var == 
            numpy.trace(A.cov*B.cov) + 
            numpy.trace(A.mean.H*A.mean*B.cov) + 
            numpy.trace(A.cov*B.mean.H*B.mean) +
            numpy.trace(A.mean.H*A.mean*B.mean.H*B.mean) - 
            (A.mean*B.mean.H)**2
        )

        #rotate traces
        self.assertTrue( Var ==
            numpy.trace(A.cov*B.cov) + 
            numpy.trace(A.mean*B.cov*A.mean.H) + 
            numpy.trace(B.mean*A.cov*B.mean.H) +
            numpy.trace(A.mean*B.mean.H*B.mean*A.mean.H) - 
            (A.mean*B.mean.H)**2
        )

        #remove traces for scalars
        self.assertTrue( Var == 
            numpy.trace(A.cov*B.cov) + 
            A.mean*B.cov*A.mean.H + 
            B.mean*A.cov*B.mean.H +
            (A.mean*B.mean.H)*(B.mean*A.mean.H) - 
            (A.mean*B.mean.H)**2
        )

        #cancel means
        self.assertTrue( Var == 
            numpy.trace(A.cov*B.cov) + 
            A.mean*B.cov*A.mean.H + 
            B.mean*A.cov*B.mean.H
        )

        #avoid covariance matrixes
        self.assertTrue( Var == 
            (A*B).trace() + 
            (B*A.mean.H).trace() + 
            (A*B.mean.H).trace()
        )

        self.assertTrue(
            A.inner(B) ==
            Mvn(
                mean= A.mean*B.mean.H,
                var= (A*B).trace() + (B*A.mean.H).trace() + (A*B.mean.H).trace()
            )
        )

        self.assertTrue( A.inner(B) == B.inner(A) )


class outerTester(myTests):
    def testDerivation(self):
        A=self.A
        B=self.B

        Na = 20
        Nb = 10

        N=Na*Nb

        #get some data from A and B
        Da=A.sample(Na)
        Db=B.sample(Nb)

        #and remake the multivariates based on the samples you just took
        A=Mvn.fromData(Da)
        B=Mvn.fromData(Db)

        out = numpy.outer(Da,Db).reshape((Na,A.ndim,Nb,B.ndim))

        self.assertTrue( Matrix(numpy.outer(Da[0,:],Db[0,:])) == out[0,:,0,:] )

        result = out.mean(2).mean(0)

        self.assertTrue( numpy.outer(A.mean,B.mean) == Matrix(result))
        self.assertTrue( A.outer(B) == Matrix(result))
        self.assertTrue( B.outer(A) == Matrix(result).H)

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
