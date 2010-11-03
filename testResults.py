#dumping new pickle
**********************************************************************
File "/home/olpc/personal-projects/kalman/mvar.py", line 940, in mvar.Mvar.__and__
Failed example:
    assert (L1&L2).mean==[1,1]
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.__and__[17]>", line 1, in <module>
        assert (L1&L2).mean==[1,1]
    AssertionError
**********************************************************************
File "/home/olpc/personal-projects/kalman/mvar.py", line 424, in mvar.Mvar.transform
Failed example:
    assert A.transform(N)== (A**N).transform()
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.Mvar.transform[5]>", line 1, in <module>
        assert A.transform(N)== (A**N).transform()
    AssertionError
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
    AssertionError
**********************************************************************
File "/home/olpc/personal-projects/kalman/mvar.py", line 1550, in mvar.newBlend
Failed example:
    assert newBlend(A,B) == wiki(A,B) or flat
Exception raised:
    Traceback (most recent call last):
      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
        compileflags, 1) in test.globs
      File "<doctest mvar.newBlend[0]>", line 1, in <module>
        assert newBlend(A,B) == wiki(A,B) or flat
    AssertionError
**********************************************************************
4 items had failures:
   1 of  28 in mvar.Mvar.__and__
   1 of  10 in mvar.Mvar.transform
   1 of   1 in mvar.mooreGiven
   1 of   1 in mvar.newBlend
***Test Failed*** 4 failures.
