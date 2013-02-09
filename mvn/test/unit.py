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

import mvn.helpers as helpers

import mvn.test.fixture as fixture

fix = fixture.MvnFixture(fixture.lookup['last'])

class myTests(unittest.TestCase):
    def setUp(self):
        fix.reset()

class commuteTester(myTests):
   def testRight(self):
        self.assertTrue( fix.A+fix.B == fix.B+fix.A )
        self.assertTrue( fix.A*fix.B == fix.B*fix.A )
        self.assertTrue( fix.B-fix.A == (-fix.A)+fix.B )
        self.assertTrue( fix.A/fix.B == (fix.B**-1)*fix.A )
        self.assertTrue( fix.A & fix.B == fix.B & fix.A )
        self.assertTrue( fix.A | fix.B == fix.B | fix.A )
        self.assertTrue( fix.A ^ fix.B == fix.B ^ fix.A )

class creationTester(myTests):
    def testFromData(self):
        self.assertTrue( Mvn.fromData(fix.A) == fix.A )

        data=[1, 2, 3]
        one = Mvn.fromData(data)
        self.assertTrue(one.ndim == 1)


        data=[[1, 2, 3]]
        many=Mvn.fromData(data)
        self.assertTrue( many.mean == data )
        self.assertTrue( Matrix(many.var) == numpy.zeros )
        self.assertTrue( many.vectors == numpy.zeros )
        self.assertTrue( many.cov == numpy.zeros )

    def testFromCov(self):
        self.assertTrue(
            Mvn.fromCov(fix.A.cov, mean=fix.A.mean) == 
            fix.A
        )
        
        self.assertTrue(
            Mvn.fromCov(Matrix.zeros([0, 0])) == Mvn()   
        )

    def testZeros(self):
        n=abs(fix.N)
        Z=Mvn.zeros(n)
        self.assertTrue( Z.mean == Matrix.zeros )
        self.assertTrue( Z.var.size == 0 )
        self.assertTrue( Z.vectors.size == 0 )
        self.assertTrue( Z**-1 == Mvn.infs)

    def testInfs(self):
        n=abs(fix.N)
        inf=Mvn.infs(n)
        self.assertTrue( inf.mean == Matrix.zeros )
        self.assertTrue( inf.var.size == inf.mean.size == n )
        self.assertTrue( Matrix(inf.var) == Matrix.infs )
        self.assertTrue( inf.vectors == Matrix.eye )
        self.assertTrue( inf**-1 == Mvn.zeros )
    
    def testEye(self):
        n=abs(fix.N)
        eye=Mvn.eye(n)
        self.assertTrue(eye.mean == Matrix.zeros)
        self.assertTrue(eye.var.size == eye.mean.size == n)
        self.assertTrue(Matrix(eye.var) == Matrix.ones)
        self.assertTrue(eye.vectors == Matrix.eye)
        self.assertTrue(eye**-1 == eye)
        
    def testCopy(self):
        A2=fix.A.copy(deep = True)        
        self.assertTrue( A2 == fix.A )
        self.assertTrue( A2 is not fix.A )
        self.assertTrue( A2.mean is not fix.A.mean )

        fix.A.copy(fix.B,deep= True)
        self.assertTrue( fix.B == fix.A )
        self.assertTrue( fix.B is not fix.A )
        self.assertTrue( fix.A.mean is not fix.B.mean )

        A2=fix.A.copy(deep= False)        
        self.assertTrue( A2 == fix.A )
        self.assertTrue( A2 is not fix.A )
        self.assertTrue( A2.mean is fix.A.mean )

        fix.A.copy(fix.B,deep= False)
        self.assertTrue( fix.B == fix.A )
        self.assertTrue( fix.B is not fix.A )
        self.assertTrue( fix.A.mean is fix.B.mean )

class densityTester(myTests):
    def testDensity(self):
        """
        Another way to think of the & operation is as doing a 
        pointwize product of the probability (densities) and then 
        normalizing the total probability of 1. 

        But it works in both directions, blend and un-blend
        is like multiply and divide.
        """
        if not (fix.A.flat or fix.B.flat):
            #remember you can undo a blend.
            self.assertTrue((~fix.B) & fix.A & fix.B == fix.A)

            #setup
            AB  = fix.A &  fix.B
            A_B = fix.A & ~fix.B
    
            locations = AB.getX()

            # A&B == k*A.*B
            Da = fix.A.density(locations)
            Db = fix.B.density(locations)

            Dab  = (AB).density(locations)

            ratio = Dab/(Da*Db)
            
            self.assertTrue(Matrix(ratio) == ratio[0])

            # A&(~B) == k*A./B
            Da_b = (A_B).density(locations)
            ratio = Da_b/(Da/Db)
            self.assertTrue(Matrix(0) == ratio.var())

            #log
            Ea = fix.A.entropy(locations)
            Eb = fix.B.entropy(locations)

            Eab  = (AB).entropy(locations)
            delta = Eab-(Ea+Eb)
            self.assertTrue(Matrix(0) == delta.var())

            Ea_b = (A_B).entropy(locations)
            delta = Ea_b-(Ea-Eb)
            self.assertTrue(Matrix(0) == delta.var())


    def testDensity2(self):
        data = fix.A.sample([5, 5])
        self.assertTrue(
            Matrix(fix.A.density(data)) == 
            numpy.exp(-fix.A.entropy(data))
        )

class equalityTester(myTests):
    def testEq(self):
        # always equal if same object
        self.assertTrue( fix.A == fix.A )
        # still equal if copy
        self.assertTrue( fix.A == fix.A.copy() )
        self.assertTrue( fix.A is not fix.A.copy() )
    
    def testCosmetic(self):
        self.assertTrue( fix.A == fix.A.square() )
        self.assertTrue( fix.A == fix.A.copy() )
        self.assertTrue( fix.A == fix.A.inflate() )
        self.assertTrue( fix.A == fix.A.inflate().squeeze() )

    def testInf(self):
        self.assertTrue(
             Mvn(mean= [1, 0, 0], vectors= [1, 0, 0], var= numpy.inf) ==
             Mvn(mean= [0, 0, 0], vectors= [1, 0, 0], var= numpy.inf)
        )
        
    def testNot(self):
        self.assertTrue( fix.A != fix.B )

class signTester(myTests):
    def testPlus(self):
        self.assertTrue( fix.A == +fix.A )
        self.assertTrue( fix.A == ++fix.A )
        self.assertTrue( fix.A is not +fix.A )
        self.assertTrue( fix.A+fix.A == 2*fix.A )
        self.assertTrue( fix.A+fix.A+fix.A == 3*fix.A )

    def testMinus(self):
        self.assertTrue( -fix.A == -1*fix.A )
        self.assertTrue( fix.A == --fix.A )
        self.assertTrue( fix.A--fix.B == fix.A+fix.B )
        self.assertTrue( fix.A-fix.B == fix.A+(-fix.B) )

    def testPos(self):
        self.assertTrue( fix.A == +fix.A == ++fix.A )
    
    def testNeg(self):
        self.assertTrue( -fix.A == -1*fix.A )
        self.assertTrue( fix.A == -1*-1*fix.A )

    def testAdd(self):
        self.assertTrue( (fix.A+fix.A) == (2*fix.A) )

        self.assertTrue( (fix.A+fix.A).mean == (2*fix.A).mean )
        self.assertTrue( (fix.A+fix.A).mean == 2*fix.A.mean )

        n=abs(fix.N)
        self.assertTrue( 
            sum(itertools.repeat(fix.A,n),Mvn.zeros(fix.A.ndim)) == 
            fix.A*n 
        )

        self.assertTrue( 
            sum(itertools.repeat(-fix.A,n),Mvn.zeros(fix.A.ndim)) == 
            fix.A*(-n) 
        )
        

        self.assertTrue( fix.A+fix.B == Mvn(
            mean=fix.A.mean+fix.B.mean,
            vectors=numpy.vstack([fix.A.vectors,fix.B.vectors]),
            var = numpy.concatenate([fix.A.var,fix.B.var]),
        ))

        self.assertTrue( (fix.A+fix.A).mean == (2*fix.A).mean )
        self.assertTrue( (fix.A+fix.A).mean == 2*fix.A.mean )

        self.assertTrue( (fix.A+fix.B).mean == fix.A.mean+fix.B.mean )
        self.assertTrue( (fix.A+fix.B).cov == fix.A.cov+fix.B.cov )

    def testSub(self):
        self.assertTrue( fix.B+(-fix.A) == fix.B+(-1)*fix.A == fix.B-fix.A )
        self.assertTrue( (fix.B-fix.A)+fix.A == fix.B )

        self.assertTrue( 
            fix.A-fix.A == 
            Mvn(mean= numpy.zeros_like(fix.A.mean)) 
        )
        self.assertTrue( (fix.A-fix.B)+fix.B == fix.A )
        self.assertTrue( (fix.A-fix.B).mean == fix.A.mean - fix.B.mean )
        self.assertTrue( (fix.A-fix.B).cov== fix.A.cov - fix.B.cov )

        self.assertTrue( 
            (fix.A+fix.B*(-fix.E)).mean == 
            fix.A.mean - fix.B.mean 
        )
        self.assertTrue( (fix.A+fix.B*(-fix.E)).cov == fix.A.cov + fix.B.cov )

        self.assertTrue( fix.A-fix.B == -(fix.B-fix.A) )
        self.assertTrue( fix.A+(fix.B-fix.B) == fix.A )


class productTester(myTests):
    def testMulTypes(self):
        self.assertTrue( isinstance(fix.A*fix.B,Mvn) )
        self.assertTrue( isinstance(fix.A*fix.M,Mvn) )
        self.assertTrue( isinstance(fix.M.T*fix.A,Matrix) )
        self.assertTrue( isinstance(fix.A*fix.K1,Mvn) )
        self.assertTrue( isinstance(fix.K1*fix.A,Mvn) )


    def testMul(self):
        self.assertTrue( fix.A**2 == fix.A*fix.A )

    def testMvnMul(self):
        self.assertTrue( 
            (fix.A*fix.B).cov == 
            (fix.A*fix.B.transform()+fix.B*fix.A.transform()).cov/2 
        )

        if not (fix.A.flat or fix.B.flat): 
            self.assertTrue( 
                (fix.A*fix.B).mean == 
                (fix.A*fix.B.transform()+fix.B*fix.A.transform()).mean/2 
            )

        self.assertTrue( fix.M.T*fix.B == fix.M.T*fix.B.transform() )
        if not fix.A.flat:
            self.assertTrue( fix.A**2 == fix.A*fix.A.transform() )

        self.assertTrue( fix.A*(fix.B**0+fix.B**0) == fix.A*(2*fix.B**0) )
        self.assertTrue( 
            (fix.A*fix.B**0 + fix.A*fix.B**0).cov == 
            (2*fix.A*fix.B**0).cov 
        )

        self.assertTrue( fix.A*fix.A == fix.A**2 )
        self.assertTrue( fix.A*fix.B == fix.B*fix.A )
        if not fix.A.flat:
            self.assertTrue( fix.A*fix.A == fix.A*fix.A.transform() )


    def testScalarMul(self):
        self.assertTrue( fix.A+fix.A == 2*fix.A )
        self.assertTrue( (2*fix.A).mean == 2*fix.A.mean )
        self.assertTrue( (2*fix.A.cov) == 2*fix.A.cov )
    
        self.assertTrue( fix.A*(fix.K1+fix.K2) == fix.A*fix.K1+fix.A*fix.K2 )

        self.assertTrue( (fix.A*(fix.K1*fix.E)).mean == fix.K1*fix.A.mean )
        self.assertTrue( 
            (fix.A*(fix.K1*fix.E)).cov == fix.A.cov*abs(fix.K1)**2 )
        self.assertTrue( (fix.K1*fix.A)*fix.K2 == fix.K1*(fix.A*fix.K2) )

        self.assertTrue( 
            (2*fix.B**0).transform() == sqrt(2)*(fix.B**0).transform() 
        )
        
        self.assertTrue( fix.A*fix.K1 == fix.K1*fix.A )

        self.assertTrue( (fix.A*fix.K1).mean == fix.K1*fix.A.mean )
        self.assertTrue( (fix.A*fix.K1).cov == (fix.A.cov)*fix.K1 )

        self.assertTrue( (fix.A*(fix.E*fix.K1)).mean == fix.A.mean*fix.K1 )
        self.assertTrue( 
            (fix.A*(fix.E*fix.K1)).cov ==
            (fix.E*fix.K1).H*fix.A.cov*(fix.E*fix.K1) 
        )


    def testMixedMul(self):
        self.assertTrue( fix.K1*fix.A*fix.M == fix.A*fix.K1*fix.M )
        self.assertTrue( fix.K1*fix.A*fix.M == fix.A*fix.M*fix.K1 )

    def testMatrixMul(self):
        self.assertTrue( (fix.A*fix.M)*fix.M2.H == fix.A*(fix.M*fix.M2.H) )
        self.assertTrue( (fix.M*fix.M2.H)*fix.A == fix.M*(fix.M2.H*fix.A) )

        self.assertTrue( (fix.A*fix.M).cov == fix.M.H*fix.A.cov*fix.M )
        self.assertTrue( (fix.A**2).transform() == fix.A.cov )
        
        self.assertTrue( fix.A*fix.E == fix.A )
        self.assertTrue( (-fix.A)*fix.E == -fix.A )

        self.assertTrue( (fix.A*fix.M).cov == fix.M.H*fix.A.cov*fix.M )
        self.assertTrue( (fix.A*fix.M).mean == fix.A.mean*fix.M )

    def testDiv(self):
        self.assertTrue( fix.A/fix.B == fix.A*fix.B**(-1) )
        
        m=fix.M*fix.M2.T
        self.assertTrue( fix.A/m == fix.A*(m**(-1)) )
        self.assertTrue( fix.A/fix.K1 == fix.A*(fix.K1**(-1)) )
        self.assertTrue( fix.K1/fix.A == fix.K1*(fix.A**(-1)) )
        self.assertTrue( fix.M.H/fix.A == fix.M.H*(fix.A**(-1)) )


class propertyTester(myTests):
    def testNdim(self):
        self.assertTrue( fix.A.ndim == fix.A.mean.size )

    def testShape(self):
        self.assertTrue( fix.A.vectors.shape == fix.A.shape )
        self.assertTrue( (fix.A.var.size,fix.A.mean.size) == fix.A.shape )
        self.assertTrue( fix.A.shape[1] == fix.A.ndim )

    def testCov(self):
        self.assertTrue( 
            fix.A.vectors.H*numpy.diagflat(fix.A.var)*fix.A.vectors == 
            fix.A.cov 
        )
        self.assertTrue( fix.A.transform()**2 == abs(fix.A).cov )
        self.assertTrue( fix.A.transform(2) == abs(fix.A).cov )
        if not(fix.A.flat and fix.N<0):
            self.assertTrue( 
                fix.A.transform()**fix.N == 
                fix.A.transform(fix.N) 
            )
            
    def testCov2(self):
        self.assertTrue( fix.A.cov == (fix.A**2).transform() )
        self.assertTrue( fix.A.cov == fix.A.transform()*fix.A.transform() )
        self.assertTrue( fix.A.cov == fix.A.transform()**2 )
        self.assertTrue( fix.A.cov == fix.A.transform(2) )
        self.assertTrue( 
            (
                fix.A*fix.B.transform() + 
                fix.B*fix.A.transform()
            ).cov/2 == 
            (fix.A*fix.B).cov 
        )

    def testScaled(self):
        self.assertTrue( fix.A.scaled.H*fix.A.scaled == abs(fix.A).cov )
        self.assertTrue( Matrix(helpers.mag2(fix.A.scaled)) == fix.A.var )
        self.assertTrue( fix.A.vectors.H*fix.A.scaled == fix.A.transform() )

    def testVectors(self):
        if fix.A.shape[0] == fix.A.shape[1]:
            self.assertTrue( fix.A.vectors.H*fix.A.vectors == Matrix.eye )
        else:
            a = fix.A.inflate()
            self.assertTrue( a.vectors.H*a.vectors == Matrix.eye )
        
        self.assertTrue(fix.A.vectors*fix.A.vectors.H == Matrix.eye)
        self.assertTrue((fix.A*fix.M).cov == fix.M.H*fix.A.cov*fix.M)
        self.assertTrue(
            (fix.A*fix.M).vectors*(fix.A*fix.M).vectors.H == 
            Matrix.eye
        )


    def testTransform(self):
        self.assertTrue( fix.A.transform() == fix.A.transform(1) )
        self.assertTrue( fix.A.transform() == fix.A.scaled.H*fix.A.vectors )


class mergeTester(myTests):
    def testStack(self):
        fix.AB= Mvn.stack(fix.A,fix.B)
        self.assertTrue( fix.AB[:,:fix.A.ndim] == fix.A )
        self.assertTrue( fix.AB[:,fix.A.ndim:] == fix.B )
        self.assertTrue( Mvn.stack(Mvn.infs(2),Mvn.infs(5)) == Mvn.infs(7) )
        self.assertTrue( Mvn.stack(Mvn.zeros(2),Mvn.zeros(5)) == Mvn.zeros(7) )
        

class powerTester(myTests):
    def testIntPowers(self):
        N = abs(fix.N)
        self.assertTrue( fix.A.transform(N) == (fix.A**N).transform() )
        self.assertTrue( fix.A.transform(N) == fix.A.transform()**N )

        N = -abs(fix.N)
        if not fix.A.flat:
            self.assertTrue( fix.A.transform(N) == (fix.A**N).transform() )
            self.assertTrue( fix.A.transform(N) == fix.A.transform()**N )


    def testMorePowers(self):
        self.assertTrue( 
            (fix.A**fix.K1).transform()**2 == 
            fix.A.transform(fix.K1)**2 
        )

        self.assertTrue( fix.A**fix.K1*fix.A**fix.K2 == fix.A**(fix.K1+fix.K2))
        self.assertTrue( fix.A**fix.K1/fix.A**fix.K2 == fix.A**(fix.K1-fix.K2))

        self.assertTrue( fix.A*fix.A**fix.K2 == fix.A**(1+fix.K2))
        self.assertTrue( fix.A/fix.A**fix.K2 == fix.A**(1-fix.K2))

        self.assertTrue( fix.A**fix.K1*fix.A == fix.A**(fix.K1+1))
        self.assertTrue( fix.A**fix.K1/fix.A == fix.A**(fix.K1-1))
 

    def testZeroPow(self):
        self.assertTrue( fix.A**0*fix.A == fix.A )
        self.assertTrue( fix.A*fix.A**0 == fix.A )
        self.assertTrue( Matrix((fix.A**0).var) == numpy.ones )

    def testZeroFlat(self):
        if not fix.A.flat:
            self.assertTrue( fix.A**0 == fix.A**(-1)*fix.A )
            self.assertTrue( fix.A**0 == fix.A*fix.A**(-1) )
            self.assertTrue( fix.A**0 == fix.A/fix.A )

            self.assertTrue( 
                (fix.A**0).mean == 
                fix.A.mean*(fix.A**-1).transform() 
            )

            self.assertTrue( 
                (fix.A**0).mean == 
                fix.A.mean*fix.A.transform(-1) 
            )


    def testOnePow(self):
        self.assertTrue( fix.A == fix.A**1 )
        self.assertTrue( -fix.A == (-fix.A)**1 )

        if not fix.A.flat:
            self.assertTrue( fix.A == (fix.A**-1)**-1 )

    def testRealPow(self):
        self.assertTrue( fix.A*fix.A == fix.A**2 )
        self.assertTrue( fix.A/fix.A**-1 == fix.A**2 )
        
        self.assertTrue( 
            fix.A.mean*fix.A.transform(0) == 
            ((fix.A**-1)**-1).mean 
        )

        k1 = fix.K1
        k2 = fix.K2
        self.assertTrue( (fix.A**k1).transform() == fix.A.transform(k1) )

        self.assertTrue( (fix.A**k1)*(fix.A**k2) == fix.A**(k1+k2) )
        self.assertTrue( fix.A**k1/fix.A**k2 == fix.A**(k1-k2) )

        if not fix.A.flat:
            self.assertTrue( 
                fix.A**k1 == (
                    fix.A*fix.A.transform(k1-1) + 
                    Mvn(mean=fix.A.mean-fix.A.mean*fix.A.transform(0))
                ))
                
class widthTester(myTests):
    def testWidth(self):
        self.assertTrue(
            Matrix([fix.A[:,n].var[0] for n in range(fix.A.ndim)]) == 
            fix.A.width()**2
        )

        self.assertTrue(
            Matrix(fix.A.corr.diagonal()) == 
            Matrix.ones
        )

        norm = fix.A/fix.A.width()
        self.assertTrue(norm.corr == norm.cov)
        self.assertTrue(
            Matrix([norm[:,n].var[0] for n in range(norm.ndim)]) == 
            Matrix.ones
        )
        
        self.assertTrue(
            Matrix((fix.A**0).var) ==
            Matrix.ones
        )
        
        data = fix.A.sample(100)
        a = Mvn.fromData(data)
        self.assertTrue(Matrix(numpy.std (data,0)) == a.width()   )    
        self.assertTrue(Matrix(numpy.var (data,0)) == a.width()**2)
        self.assertTrue(Matrix(numpy.mean(data,0)) == a.mean      )

class linalgTester(myTests):
    def testTrace(self):
        self.assertTrue( 
            Matrix(numpy.trace(fix.A.transform(0))) == 
            fix.A.shape[0] 
        )

        self.assertTrue( Matrix(fix.A.trace()) == fix.A.var.sum() )
        self.assertTrue( Matrix(fix.A.trace()) == numpy.trace(fix.A.cov) )

    def testDet(self):
        self.assertTrue( Matrix(fix.A.det()) == numpy.linalg.det(fix.A.cov) )
        self.assertTrue( Matrix(fix.A.det()) == 
             0 if 
             fix.A.shape[0] != fix.A.shape[1] else 
             numpy.prod(fix.A.var)
        )

    def testDist2(self):
        if not fix.A.flat:
            self.assertTrue( 
                Matrix((fix.A**0).dist2(numpy.zeros((1,fix.ndim)))) == 
                helpers.mag2((fix.A**0).mean) 
            )

    def testSquare(self):
        vectors = Matrix(helpers.ascomplex(numpy.random.randn(
            numpy.random.randint(1,10),numpy.random.randint(1,10),2
        )))
        cov = vectors.H*vectors
        Xcov = vectors*vectors.H 
        (Xval,Xvec) = numpy.linalg.eigh(Xcov)
        vec = Xvec.H*vectors
        self.assertTrue( vec.H*vec == cov )

class givenTester(myTests):            
    def testGivenScalar(self):
        a = fix.A.given(dims= 0,value= 1)
        self.assertTrue( a.mean[:, 0] == 1 )
        self.assertTrue( a.vectors[:, 0] == numpy.zeros )

        a=fix.A.copy(deep= True)
        a[:, 0] = 1
        self.assertTrue( a == fix.A.given(dims= 0, value= 1) )


    def testGivenLinear(self):
        L1 = Mvn(mean= [0, 0], vectors=[[1, 1],[1, -1]], var=[numpy.inf, 0.5])
        L2=Mvn(mean=[1, 0], vectors=[0, 1], var=numpy.inf) 
        self.assertTrue( L1.given(dims=0, value=1) == L1&L2 )
        self.assertTrue( (L1&L2).mean == [1, 1] )
        self.assertTrue( (L1&L2).cov == [[0, 0], [0, 2]] )

    def testGivenMvn(self):
        Y=Mvn(mean=[0, 1], vectors=Matrix.eye, var=[numpy.inf, 1])
        X=Mvn(mean=[1, 0], vectors=Matrix.eye, var=[1, numpy.inf])
        x=Mvn(mean=1, var=1)
        self.assertTrue( Y.given(dims=0, value=x) == X&Y )

    def testGivenVector(self):
        self.assertTrue( 
            givenVector(fix.A, dims=0, value=1) == 
            fix.A.given(dims=0, value=1) 
        )

class chainTester(myTests):
    def testBasic(self):
        self.assertTrue( fix.A.chain() == fix.A*numpy.hstack([fix.E, fix.E]) ) 

        self.assertTrue( 
            fix.A.chain(transform=fix.M) ==
            fix.A*numpy.hstack([fix.E, fix.M])
        )

    def testMoore(self):
        self.assertTrue( fix.A.chain(fix.B) == mooreChain(fix.A, fix.B) )

        b=fix.B*fix.M
        self.assertTrue( fix.A.chain(b, fix.M) == mooreChain(fix.A, b, fix.M) )

    def testStacks(self):
        dataA=fix.A.sample(100)

        a=Mvn.fromData(dataA)

        #a and a are correlated
        self.assertTrue(
            a.chain()==
            Mvn.fromData(numpy.hstack([dataA, dataA]))
        )        
        #a and a*M are corelated        
        self.assertTrue(
            a.chain(transform=fix.M) == 
            dataA*numpy.hstack([fix.E, fix.M])
        )

        self.assertTrue( 
            a.chain(transform= fix.M) == 
            Mvn.fromData(numpy.hstack([dataA, dataA*fix.M]))
        )

        self.assertTrue(
            a.chain(fix.B*fix.M,fix.M) == 
            a.chain(transform= fix.M)+Mvn.stack(Mvn.zeros(a.ndim), fix.B*fix.M)
        )

        

    def testAnd(self):
        """
        __and__ is a shortcut across mvn.chain and mvn.given
        this is to show the relationship

        I haven't figured yet out how a the 'transform' parameter to chain works 
        with __and__, it probably involves the psudo-inverse of the transform. 

        I think the answer is on the wikipedia kalman-filtering page
        """

        measurment = fix.B.mean
        sensor = fix.B.copy()
        sensor.mean = Matrix.zeros(sensor.mean.shape)

        joint = fix.A.chain(sensor)
        measured = joint.copy()
        measured[:, fix.ndim:] = measurment

        self.assertTrue(measured[:, :fix.ndim] == fix.A&fix.B)


class inversionTester(myTests):
    def testAbs(self):
        self.assertTrue( (fix.A.var >= 0).all() )
        self.assertTrue( abs(fix.A) == abs(~fix.A) )

    def testNeg(self):
        IA = fix.A.copy()
        IA.var = -IA.var
        self.assertTrue( IA == ~fix.A )

        self.assertTrue( Matrix((~fix.A).var) == (-fix.A).var )
        self.assertTrue( Matrix((~fix.A).var) == -(fix.A.var) )

    def testInvariant(self):
        self.assertTrue( (~fix.A).mean == fix.A.mean )
        self.assertTrue( (~fix.A).vectors == fix.A.vectors )
        self.assertTrue( (~fix.A).cov == (-fix.A).cov )
        self.assertTrue( (~fix.A).cov == -(fix.A.cov) )
        
    def testDoubleNegative(self):
        self.assertTrue( ~~fix.A == fix.A )
        self.assertTrue( ~(~fix.A&~fix.B) == fix.A & fix.B )
        self.assertTrue( (~fix.A & ~fix.B) == ~(fix.A & fix.B) )


    def testParadoxes(self):
        self.assertTrue( 
            (fix.A & ~fix.A) == 
            Mvn(mean= fix.A.mean, vectors= fix.A.vectors, var= Matrix.infs) 
        )
        
        self.assertTrue( (fix.A & ~fix.A)*fix.A.vectors.H == Mvn.infs )

        self.assertTrue(  
            fix.A & (fix.B & ~fix.B) == 
            fix.A & Mvn(
                mean= fix.B.mean, 
                vectors= fix.B.vectors, 
                var= Matrix.infs
            )
        )

        if not fix.B.flat:        
            self.assertTrue( fix.A == fix.A & (fix.B & ~fix.B) )
        
        self.assertTrue( (fix.A&~fix.B) & fix.B == (fix.A&fix.B) & ~fix.B )

        self.assertTrue( (fix.A&fix.B) & ~fix.B == fix.A & (fix.B&~fix.B) )

        self.assertTrue( not numpy.isfinite((fix.A & ~fix.A).var).any() )

        P = fix.A.copy()
        P.var = P.var/0.0
        self.assertTrue( P == (fix.A & ~fix.A) )


    def testPow(self):
        self.assertTrue(
           ( fix.A)**(-1) + (~fix.A)**(-1) == 
           Mvn.zeros
        )
        
        self.assertTrue(
           (( fix.A)**(-1) + (~fix.A)**(-1))**-1 == 
           Mvn.zeros(fix.A.ndim)**-1
        )    

class blendTester(myTests):
    def testCommutativity(self):
        self.assertTrue( fix.A & fix.B == fix.B & fix.A)
        
    def testSelf(self):
        self.assertTrue( (fix.A & fix.A).cov == fix.A.cov/2)
        self.assertTrue( (fix.A & fix.A).mean == fix.A.mean)
        
        
    def testNotFlat(self):
        if not (fix.A.flat or fix.B.flat):
            self.assertTrue( fix.A & fix.B == 1/(1/fix.A+1/fix.B))
            
            self.assertTrue( 
                fix.A & -fix.A == 
                Mvn(mean= numpy.zeros(fix.ndim))**-1
            )
    
            self.assertTrue( 
                fix.A & ~fix.A == 
                Mvn(mean= numpy.zeros(fix.ndim))**-1
            )
            self.assertTrue( fix.A & fix.B == wiki(fix.A,fix.B))
               
            self.assertTrue( fix.A**-1 == fix.A*fix.A**-2)
            
            self.assertTrue( 
                fix.A & fix.B == 
                (fix.A*fix.A**-2+fix.B*fix.B**-2)**-1
            )

            D = fix.A*(fix.A.cov)**(-1) + fix.B*(fix.B.cov)**(-1)
            self.assertTrue( wiki(fix.A,fix.B) == D*(D.cov)**(-1))
            self.assertTrue( fix.A & fix.B == wiki(fix.A,fix.B))

        if not (fix.A.flat or fix.B.flat or fix.C.flat):
            abc=numpy.random.permutation([fix.A, fix.B, fix.C])
            self.assertTrue( fix.A & fix.B & fix.C == helpers.parallel(*abc))
            self.assertTrue( 
                fix.A & fix.B & fix.C == 
                reduce(operator.and_, abc)
            )
    
            self.assertTrue( 
                (fix.A & fix.B) & fix.C == 
                fix.A & (fix.B & fix.C)
            )


    def testKnownValues1(self):
        L1=Mvn(mean= [1, 0], vectors= [0, 1], var= numpy.inf)
        L2=Mvn(mean= [0, 1], vectors= [1, 0], var= numpy.inf) 
        self.assertTrue( (L1&L2).mean == [1, 1])
        self.assertTrue( (L1&L2).var.size == 0)

    def testKnownValues2(self):
        L1=Mvn(mean= [0, 0], vectors= [1, 1], var= numpy.inf)
        L2=Mvn(mean= [0, 1], vectors= [1, 0], var= numpy.inf) 
        self.assertTrue( (L1&L2).mean == [1, 1])
        self.assertTrue( (L1&L2).var.size == 0)

    def testKnownValues3(self):
        L1=Mvn(mean= [0, 0], vectors= Matrix.eye, var=[1, 1])
        L2=Mvn(mean= [0, 1], vectors= [1, 0], var= numpy.inf) 
        self.assertTrue( (L1&L2).mean == [0, 1] )
        self.assertTrue( (L1&L2).var == 1 )
        self.assertTrue( (L1&L2).vectors == [1, 0] )

class quadTester(myTests):
    def testDerivation(self):
        Na = 25

        #get some data from A
        Da = Matrix(fix.A.sample(Na))

        #and remake the multivariates
        A = Mvn.fromData(Da)

        # take all the dot products
        dots = (numpy.array(Da)**2).sum(1)
        self.assertTrue( Matrix(dots) == numpy.diag(Da*Da.H) )

        Mean = Matrix(dots.mean())
        Var = Matrix(dots.var())


        self.assertTrue( Mean == numpy.trace(Da*Da.H)/Na )
        self.assertTrue( Mean == numpy.trace(Da.H*Da/Na) )
        self.assertTrue( Mean == (Da*Da.H).diagonal().mean() )

        self.assertTrue( A.cov+A.mean.H*A.mean == (Da.H*Da)/Na )

        self.assertTrue( Mean == numpy.trace(A.mean.H*A.mean + A.cov) )

        self.assertTrue( 
            Mean == 
            numpy.trace(A.mean.H*A.mean)+numpy.trace(A.cov) 
        )

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
        self.assertTrue( 
            Var == Mean**2 - 2*Mean*dots.mean() + (dots**2).mean() 
        )

        #but Mean == dot.mean(), so
        self.assertTrue( Var == (dots**2).mean() - Mean**2 )

        self.assertTrue( Var == (dots**2).sum()/Na - Mean**2 )

        self.assertTrue( Var == ((Da*Da.H).diagonal()**2).sum()/Na - Mean**2 )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())*
            Matrix((Da*Da.H).diagonal()).H/Na-
            Mean**2
        )

        self.assertTrue( 
            Mean ==
            (Matrix((Da*Da.H).diagonal())*
            Matrix.ones((Na,1))/Na)
        )

        self.assertTrue( 
            Mean**2 == 
            (Matrix((Da*Da.H).diagonal())*
            Matrix.ones((Na,1))/Na)**2
        )

        self.assertTrue( 
            Mean**2 == 
            Matrix((Da*Da.H).diagonal() *
            Matrix.ones((Na,1))/Na) *
            Matrix((Da*Da.H).diagonal() *
            Matrix.ones((Na,1))/Na)
        )

        self.assertTrue( 
            Mean**2 == 
            Matrix((Da*Da.H).diagonal()) *
            Matrix.ones((Na, 1))*Matrix.ones((1, Na))/Na**2 * 
            Matrix((Da*Da.H).diagonal()).H 
        )

        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())*
            Matrix((Da*Da.H).diagonal()).H/Na 
            -
            Matrix((Da*Da.H).diagonal())*
            Matrix.ones((Na, 1))*Matrix.ones((1, Na))/Na**2* 
            Matrix((Da*Da.H).diagonal()).H
        )
            
        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())*
            Matrix((Da*Da.H).diagonal()).H/Na 
            -
            (Matrix((Da*Da.H).diagonal())*
            Matrix((Da*Da.H).diagonal()).H.sum()).sum()/Na/Na
        )

        self.assertTrue( 
            Var ==
            Matrix((Da*Da.H).diagonal())/Na *
            Matrix((Da*Da.H).diagonal()).H
            -
            Matrix((Da*Da.H).diagonal())/Na *
            (numpy.trace(Da*Da.H) *
            Matrix.ones((Na,1)))/Na 
        )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())/Na *
            (
                Matrix((Da*Da.H).diagonal()).H
                -
                (numpy.trace(Da*Da.H) *
                Matrix.ones((Na,1)))/Na
            )
        )

        self.assertTrue( 
            Var == 
            Matrix((Da*Da.H).diagonal())/Na *
            (
                Matrix((Da*Da.H).diagonal()).H
                -Mean
            )
        )

        #there's a connection in between here that I don't understand   

        #wiki: this is the Reference value
        wVar = 2*numpy.trace(A.cov*A.cov)+4*A.mean*A.cov*A.mean.H

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov*
                A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) 
            + 
            4*numpy.trace(
                A.mean.H*A.mean*
                A.vectors.H*numpy.diagflat(A.var)*A.vectors
            )
        )

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov+
                A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) 
            + 
            numpy.trace(
                4*A.mean *
                A.vectors.H*numpy.diagflat(A.var)*A.vectors *
                A.mean.H
            )
        )

        self.assertTrue( 
            wVar == 
            2*numpy.trace(
                A.cov
                *A.vectors.H*numpy.diagflat(A.var)*A.vectors
            ) + 
            numpy.trace(
                4*A.mean *
                A.vectors.H*numpy.diagflat(A.var)*A.vectors *
                A.mean.H
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
            A.quad() ==
            Mvn(
                mean= A.mean*A.mean.H + A.trace(),
                var= 2*(A*A).trace()+4*(A*A.mean.H).trace()
            )
        )

class innerTester(myTests):
    def testDerivation(self):
        A = fix.A
        B = fix.B

        Na = 20
        Nb = 10

        N = Na*Nb

        #get some data from A and B
        Da = Matrix(A.sample(Na))
        Db = Matrix(B.sample(Nb))

        #and remake the multivariates based on the samples you just took
        A = Mvn.fromData(Da)
        B = Mvn.fromData(Db)

        # take every possible combination of dot products
        dot = numpy.array(Da*Db.H)

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
                2*numpy.multiply(Mean, dot) + dot**2 
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
                (A.cov+A.mean.H*A.mean)*
                (B.cov+B.mean.H*B.mean)
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
            ) - (
                A.mean*B.mean.H)**2 
            )

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
        A = fix.A
        B = fix.B

        Na = 20
        Nb = 10

        N=Na*Nb

        #get some data from A and B
        Da = A.sample(Na)
        Db = B.sample(Nb)

        #and remake the multivariates based on the samples you just took
        A = Mvn.fromData(Da)
        B = Mvn.fromData(Db)

        out = numpy.outer(Da, Db).reshape((Na, A.ndim, Nb, B.ndim))

        self.assertTrue( 
            Matrix(numpy.outer(Da[0, :], Db[0, :])) == 
            out[0, :, 0, :] 
        )

        result = out.mean(2).mean(0)

        self.assertTrue( numpy.outer(A.mean, B.mean) == Matrix(result))
        self.assertTrue( A.outer(B) == Matrix(result))
        self.assertTrue( B.outer(A) == Matrix(result).H)

def wiki(P, M):
    """
    :param P:
    :param M:
        
    Direct implementation of the wikipedia blending algorithm
    """
    yk = M.mean-P.mean
    Sk = P.cov+M.cov
    Kk = P.cov*Sk.I
    
    return Mvn.fromCov(
        mean= (P.mean + yk*Kk.H),
        cov= (Matrix.eye(P.ndim)-Kk)*P.cov
    )

def givenVector(self, dims, value):
    """
    :param dims:
    :param value:
    
    direct implementation of the "given" algorithm in
    Andrew moore's data-mining/gussian slides

    >>> assert givenVector(A,dims=0,value=1)==A.given(dims=0,value=1)
    """
    fixed = helpers.binindex(dims, self.ndim)
    if fixed.all():
        return Mvn.fromData(value)

    free =~ fixed

    Mu = self[:, free]
    Mv = self[:, fixed]
    #TODO: cleanup
    u = self.vectors[:, free]
    v = self.vectors[:, fixed]

    uv = numpy.multiply(u.H, self.var)*v

    result = Mu-(Mv-value)**-1*uv.H

    #create the mean, for the new object,and set the values of interest
    mean = numpy.zeros([1, self.ndim], dtype=result.mean.dtype)
    mean[:, fixed] = value
    mean[:, free] = result.mean

    #create empty vectors for the new object
    vectors=numpy.zeros([
        result.shape[0],
        self.ndim,
    ],result.vectors.dtype)
    vectors[:, fixed] = 0
    vectors[:, free] = result.vectors
    
    return type(self)(
        mean= mean,
        vectors= vectors,
        var= result.var
    )


def mooreChain(self, sensor, transform=None):
    """
    :param sensor:
    :param transform:
        
    given a distribution of actual values and an Mvn to act as a sensor 
    this method returns the joint distribution of real and measured values

    the, optional, transform parameter describes how to transform from actual
    space to sensor space
    """

    if transform is None:
        transform = Matrix.eye(self.ndim)

    T = (self*transform+sensor)
    vv = self.cov        

    return type(self).fromCov(
        mean = numpy.hstack([self.mean, T.mean]),
        cov = numpy.vstack([
            numpy.hstack([vv, vv*transform]),
            numpy.hstack([(vv*transform).H, T.cov]),
        ])
    )



class refereceTester(myTests):
    
    def testMooreChain(self):
        #when including a sensor, noise is added to those new dimensions

        self.assertTrue(
            fix.A.chain(fix.B) == 
            mooreChain(fix.A, fix.B)
        )
        self.assertTrue(
            fix.A.chain(fix.B*fix.M, fix.M) == 
            mooreChain(fix.A, fix.B*fix.M, fix.M)
        )


    def testWiki(self):
        if not (fix.A.flat or fix.B.flat):
            self.assertTrue( fix.A & fix.B == wiki(fix.A, fix.B) )

        #The quickest way to prove it's equivalent is by examining these:

            self.assertTrue( fix.A**-1 == fix.A*fix.A**-2 )
            self.assertTrue( 
                fix.A & fix.B == 
                (fix.A*fix.A**-2+fix.B*fix.B**-2)**-1 
            )
        
            D = fix.A*(fix.A.cov)**(-1) + fix.B*(fix.B.cov)**(-1)
            self.assertTrue( wiki(fix.A, fix.B) == D*(D.cov)**(-1) )
            assert fix.A & fix.B == wiki(fix.A, fix.B)


def getTests(fixture=None):
    testCases = [
        value for (name, value) in globals().iteritems() 
        if isinstance(value, type) and issubclass(value, myTests)
    ]
    
    if fixture is None:
        return testCases
   
    jar=cPickle.dumps(fixture)
    testCases = [
        unittest.makeSuite(
            type(tc.__name__, (tc,), {'jar':jar})
        ) for tc in testCases
    ]

    
    return testCases

