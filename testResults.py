#dumping new pickle
**********************************************************************
File "/home/olpc/personal-projects/kalman/mvar.py", line 1571, in mvar.mooreGiven
Failed example:
    assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.mooreGiven[0]>", line 1, in <module>
        assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)
      File "/home/olpc/personal-projects/kalman/mvar.py", line 1579, in mooreGiven
        vu=numpy.diagflat(self.var)*V.vectors.H*U.vectors
      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 269, in __rmul__
        return N.dot(other, self)
    ValueError: matrices are not aligned
**********************************************************************
1 items had failures:
   1 of   1 in mvar.mooreGiven
***Test Failed*** 1 failures.
