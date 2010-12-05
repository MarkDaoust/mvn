...........F.......F.......F.FFF.F.F.F.F.F.F.F.....
======================================================================
FAIL: testNot (__main__.equalityTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 93, in testNot
    assert A!=B
AssertionError

======================================================================
FAIL: testNeg (__main__.inversionTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 388, in testNeg
    assert IA == ~A
AssertionError

======================================================================
FAIL: testIntPowers (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 281, in testIntPowers
    assert A.transform(N)== (A**N).transform()
AssertionError

======================================================================
FAIL: testRealPow (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 311, in testRealPow
    assert A*A==A**2
AssertionError

======================================================================
FAIL: testRealPowers (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 285, in testRealPowers
    assert (A**K1.real).transform() == A.transform(K1.real)
AssertionError

======================================================================
FAIL: testZeroPow (__main__.powerTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 291, in testZeroPow
    assert A**0*A==A
AssertionError

======================================================================
FAIL: testMatrixMul (__main__.productTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 214, in testMatrixMul
    assert (A**2).transform() == A.cov
AssertionError

======================================================================
FAIL: testMul (__main__.productTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 162, in testMul
    assert A**2==A*A
AssertionError

======================================================================
FAIL: testMvarMul (__main__.productTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 173, in testMvarMul
    assert A**2==A*A.transform() or flat
AssertionError

======================================================================
FAIL: testCov (__main__.propertyTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 263, in testCov
    assert A.cov == (A**2).transform()
AssertionError

======================================================================
FAIL: testScaled (__main__.propertyTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 248, in testScaled
    assert Matrix(helpers.mag2(A.scaled))==A.var
AssertionError

======================================================================
FAIL: testTransform (__main__.propertyTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 260, in testTransform
    assert A.transform() == A.scaled.H*A.vectors
AssertionError

======================================================================
FAIL: testAdd (__main__.signTester)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "./testUnit.py", line 123, in testAdd
    assert numpy.array(itertools.repeat(A,N)).sum() == A*max(N,0)
  File "/home/olpc/personal-projects/kalman/mvar.py", line 784, in __eq__
    other=Mvar.fromData(other)
  File "/home/olpc/personal-projects/kalman/mvar.py", line 283, in fromData
    assert data.dtype is not numpy.dtype('object'),'not mplementd for mvars yet'
AssertionError: not mplementd for mvars yet

----------------------------------------------------------------------
Ran 51 tests in 9.899s

FAILED (failures=13)
