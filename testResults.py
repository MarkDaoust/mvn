..................................F.......F..........F...............................F.
======================================================================
FAIL: testMooreGiven (unitTests.givenTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 398, in testMooreGiven
    self.assertTrue( mvar.mooreGiven(self.A,index=0,value=1)==self.A.given(index=0,value=1)[1:] )
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1616, in mooreGiven
    mean=self.mean[0,Iu]+uv.H*vvI*(value-self.mean[0,Iv]),
  File "matrix.py", line 51, in __add__
    assert self.shape == other.shape,'can only add matrixes with the same shape'
AssertionError: can only add matrixes with the same shape

======================================================================
FAIL: testComplexPowers (unitTests.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 305, in testComplexPowers
    self.assertTrue( self.A**self.K1*self.A**self.K2 == self.A**(self.K1+self.K2))
AssertionError

======================================================================
FAIL: testNeg (unitTests.inversionTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/olpc/personal-projects/kalman/unitTests.py", line 409, in testNeg
    self.assertTrue( self.IA == ~self.A )
AssertionError

======================================================================
FAIL: Doctest: mvar.mooreGiven
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python2.5/doctest.py", line 2128, in runTest
    raise self.failureException(self.format_failure(new.getvalue()))
AssertionError: Failed doctest test for mvar.mooreGiven
  File "/home/olpc/personal-projects/kalman/mvar.py", line 1599, in mooreGiven

----------------------------------------------------------------------
File "/home/olpc/personal-projects/kalman/mvar.py", line 1604, in mvar.mooreGiven
Failed example:
    assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)[1:]
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.mooreGiven[0]>", line 1, in <module>
        assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)[1:]
      File "/home/olpc/personal-projects/kalman/mvar.py", line 1616, in mooreGiven
        mean=self.mean[0,Iu]+uv.H*vvI*(value-self.mean[0,Iv]),
      File "matrix.py", line 51, in __add__
        assert self.shape == other.shape,'can only add matrixes with the same shape'
    AssertionError: can only add matrixes with the same shape


----------------------------------------------------------------------
Ran 87 tests in 31.582s

FAILED (failures=4)
