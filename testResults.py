...............F.FFF..F.......FF..........EEEF...........F...F.F...F..F..F...........F.
======================================================================
ERROR: testGivenLinear (__main__.givenTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 375, in testGivenLinear
    self.assertTrue( L1.given(index=0,value=1) == L1&L2 )
  File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
    mean[0,index]=value.mean
ValueError: array is not broadcastable to correct shape

======================================================================
ERROR: testGivenMvar (__main__.givenTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 383, in testGivenMvar
    self.assertTrue( Y.given(index=0,value=x) == X&Y )
  File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
    mean[0,index]=value.mean
ValueError: array is not broadcastable to correct shape

======================================================================
ERROR: testGivenScalar (__main__.givenTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 363, in testGivenScalar
    a = self.A.given(index=0,value=1)
  File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
    mean[0,index]=value.mean
ValueError: array is not broadcastable to correct shape

======================================================================
FAIL: testPlus (__main__.signTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 101, in testPlus
    self.assertTrue( self.A+self.A+self.A == 3*self.A )
AssertionError

======================================================================
FAIL: testSub (__main__.signTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 142, in testSub
    self.assertTrue( self.A-self.A == Mvar(mean=numpy.zeros_like(self.A.mean)) )
AssertionError

======================================================================
FAIL: testsAdd (__main__.signTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 124, in testsAdd
    self.assertTrue( numpy.array(itertools.repeat(self.A,n)).sum() == self.A*n )
  File "/home/olpc/personal-projects/kalman/mvar.py", line 794, in __eq__
    other=Mvar.fromData(other)
  File "/home/olpc/personal-projects/kalman/mvar.py", line 283, in fromData
    assert data.dtype is not numpy.dtype('object'),'not mplementd for mvars yet'
AssertionError: not mplementd for mvars yet

======================================================================
FAIL: testComplexPowers (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 292, in testComplexPowers
    self.assertTrue( (self.A**self.K1).transform() == self.A.transform(self.K1) )
AssertionError

======================================================================
FAIL: testRealPow (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 320, in testRealPow
    self.assertTrue( self.A/self.A**-1 == self.A**2 )
AssertionError

======================================================================
FAIL: testNeg (__main__.inversionTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 397, in testNeg
    self.assertTrue( self.IA == ~self.A )
AssertionError

======================================================================
FAIL: testParadoxes (__main__.inversionTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 415, in testParadoxes
    self.assertTrue( (self.A & ~self.A) == Mvar(mean=self.A.mean, vectors=self.A.vectors, var=Matrix.infs) )
AssertionError

======================================================================
FAIL: testMooreGiven (__main__.givenTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 386, in testMooreGiven
    self.assertTrue( mooreGiven(self.A,index=0,value=1)==self.A.given(index=0,value=1)[1:] )
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1617, in mooreGiven
    mean=self.mean[0,Iu]+uv.H*vvI*(value-self.mean[0,Iv]),
  File "matrix.py", line 51, in __add__
    assert self.shape == other.shape,'can only add matrixes with the same shape'
AssertionError: can only add matrixes with the same shape

======================================================================
FAIL: Doctest: mvar.Mvar.__add__
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar.__add__
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1431, in __add__

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 1461, in mvar.Mvar.__add__
Failed example:
    assert A-A == Mvar(mean=numpy.zeros_like(A.mean))
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__add__[5]>", line 1, in <module>
        assert A-A == Mvar(mean=numpy.zeros_like(A.mean))
    AssertionError


======================================================================
FAIL: Doctest: mvar.Mvar.__invert__
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar.__invert__
  File "/home/olpc/personal-projects/kalman/mvar.py", line 867, in __invert__

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 898, in mvar.Mvar.__invert__
Failed example:
    assert (A & ~A) == Mvar(mean=A.mean, vectors=A.vectors, var=Matrix.infs)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[11]>", line 1, in <module>
        assert (A & ~A) == Mvar(mean=A.mean, vectors=A.vectors, var=Matrix.infs)
    AssertionError
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 904, in mvar.Mvar.__invert__
Failed example:
    if not B.flat
        assert A == A & (B & ~B)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[13]>", line 1
         if not B.flat
                     
    ^
     SyntaxError: invalid syntax
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 909, in mvar.Mvar.__invert__
Failed example:
    assert  A &(B & ~B) == A & Mvar(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[14]>", line 1, in <module>
        assert  A &(B & ~B) == A & Mvar(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
    AssertionError
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 912, in mvar.Mvar.__invert__
Failed example:
    assert (A&~B) & B == (A&B) & ~B
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[15]>", line 1, in <module>
        assert (A&~B) & B == (A&B) & ~B
    AssertionError
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 913, in mvar.Mvar.__invert__
Failed example:
    if not flat
       assert (A&B) & ~B == A & (B&~B) and flat
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[16]>", line 1
         if not flat
                   
    ^
     SyntaxError: invalid syntax
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 926, in mvar.Mvar.__invert__
Failed example:
    assert not numpy.isfinite((A & ~A).var).any()
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[17]>", line 1, in <module>
        assert not numpy.isfinite((A & ~A).var).any()
    AssertionError
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 930, in mvar.Mvar.__invert__
Failed example:
    assert P==(A & ~A)       
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__invert__[20]>", line 1, in <module>
        assert P==(A & ~A)
    AssertionError


======================================================================
FAIL: Doctest: mvar.Mvar.__pow__
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar.__pow__
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1061, in __pow__

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 1092, in mvar.Mvar.__pow__
Failed example:
    assert A == (A**-1)**-1 or flat
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__pow__[4]>", line 1, in <module>
        assert A == (A**-1)**-1 or flat
    NameError: name 'flat' is not defined


======================================================================
FAIL: Doctest: mvar.Mvar._scalarMul
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar._scalarMul
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1258, in _scalarMul

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 1276, in mvar.Mvar._scalarMul
Failed example:
    assert sum(itertools.repeat(A,N-1),A) == A*(N) or N<=0
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar._scalarMul[4]>", line 1, in <module>
        assert sum(itertools.repeat(A,N-1),A) == A*(N) or N<=0
    AssertionError


======================================================================
FAIL: Doctest: mvar.Mvar.dist2
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar.dist2
  File "/home/olpc/personal-projects/kalman/mvar.py", line 648, in dist2

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 653, in mvar.Mvar.dist2
Failed example:
    if A.flat:
        assert helpers.approx(
            (A**0).dist2(numpy.zeros((1,ndim))),
            helpers.mag2((A**0).mean)
        )
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.dist2[0]>", line 4, in <module>
        helpers.mag2((A**0).mean)
    AssertionError


======================================================================
FAIL: Doctest: mvar.Mvar.given
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.Mvar.given
  File "/home/olpc/personal-projects/kalman/mvar.py", line 675, in given

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 685, in mvar.Mvar.given
Failed example:
    a = A.given(index=0,value=1)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[0]>", line 1, in <module>
        a = A.given(index=0,value=1)
      File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
        mean[0,index]=value.mean
    ValueError: array is not broadcastable to correct shape
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 686, in mvar.Mvar.given
Failed example:
    assert a.mean[:,0]==1
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[1]>", line 1, in <module>
        assert a.mean[:,0]==1
    NameError: name 'a' is not defined
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 687, in mvar.Mvar.given
Failed example:
    assert a.vectors[:,0]==numpy.zeros
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[2]>", line 1, in <module>
        assert a.vectors[:,0]==numpy.zeros
    NameError: name 'a' is not defined
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 694, in mvar.Mvar.given
Failed example:
    assert L1.given(index=0,value=1) == L1&L2
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[5]>", line 1, in <module>
        assert L1.given(index=0,value=1) == L1&L2
      File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
        mean[0,index]=value.mean
    ValueError: array is not broadcastable to correct shape
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 707, in mvar.Mvar.given
Failed example:
    assert Y.given(index=0,value=x) == X&Y
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[11]>", line 1, in <module>
        assert Y.given(index=0,value=x) == X&Y
      File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
        mean[0,index]=value.mean
    ValueError: array is not broadcastable to correct shape
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 712, in mvar.Mvar.given
Failed example:
    a[0]=1
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[13]>", line 1, in <module>
        a[0]=1
      File "/home/olpc/personal-projects/kalman/mvar.py", line 752, in __setitem__
        self.copy(self.given(index,value))
      File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
        mean[0,index]=value.mean
    ValueError: array is not broadcastable to correct shape
----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 713, in mvar.Mvar.given
Failed example:
    assert a==A.given(index=0,value=1)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.given[14]>", line 1, in <module>
        assert a==A.given(index=0,value=1)
      File "/home/olpc/personal-projects/kalman/mvar.py", line 721, in given
        mean[0,index]=value.mean
    ValueError: array is not broadcastable to correct shape


======================================================================
FAIL: Doctest: mvar.mooreGiven
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.mooreGiven
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1600, in mooreGiven

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 1605, in mvar.mooreGiven
Failed example:
    assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)[1:]
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.mooreGiven[0]>", line 1, in <module>
        assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)[1:]
      File "/home/olpc/personal-projects/kalman/mvar.py", line 1617, in mooreGiven
        mean=self.mean[0,Iu]+uv.H*vvI*(value-self.mean[0,Iv]),
      File "matrix.py", line 51, in __add__
        assert self.shape == other.shape,'can only add matrixes with the same shape'
    AssertionError: can only add matrixes with the same shape


----------------------------------------------------------------------
Ran 87 tests in 22.912s

FAILED (failures=15, errors=3)
