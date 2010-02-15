from numpy import array,matrix
from mvar import Mvar
#test objects used
A= Mvar.from_attr(
    mean=
        array([-2.66598266,  8.97029925]),
    scale=
        array([[  3.53492228,   0.        ],
               [  0.        ,  12.71289827]]),
    vectors=
        array([[-0.88691825, -0.46192642],
               [-0.46192642,  0.88691825]]),
)
B= Mvar.from_attr(
    mean=
        array([ 0.17135152,  9.75437408]),
    scale=
        array([[  1.64405224,   0.        ],
               [  0.        ,  12.50475442]]),
    vectors=
        array([[-0.89031633,  0.45534254],
               [ 0.45534254,  0.89031633]]),
)
C= Mvar.from_attr(
    mean=
        array([-0.76083441,  6.06701163]),
    scale=
        array([[ 10.35876355,   0.        ],
               [  0.        ,  40.00687094]]),
    vectors=
        array([[-0.88066574,  0.47373817],
               [ 0.47373817,  0.88066574]]),
)
M= matrix([[-0.65711363,  1.2587948 ],
        [ 0.06248338, -1.2793463 ]])
K1= (0.832574999811+0j)
K2= (0.517940128271+0j)
N= 7
#**********************************************************************
#File "./mvar.py", line 811, in __main__.Mvar.__add__
#Failed example:
#    assert numpy.allclose((A-B).cov, A.cov - B.cov)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__add__[8]>", line 1, in <module>
#        assert numpy.allclose((A-B).cov, A.cov - B.cov)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 820, in __main__.Mvar.__add__
#Failed example:
#    assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__add__[11]>", line 1, in <module>
#        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 499, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose(
#        (A**0).mean,
#        dot(
#            A.mean,A.rotation.T,A.scale**-1,A.rotation
#    ))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[5]>", line 4, in <module>
#        A.mean,A.rotation.T,A.scale**-1,A.rotation
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 771, in __main__.Mvar.__rdiv__
#Failed example:
#    assert M/A == M*(A**(-1))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rdiv__[1]>", line 1, in <module>
#        assert M/A == M*(A**(-1))
#    AssertionError
#**********************************************************************
#3 items had failures:
#   2 of  12 in __main__.Mvar.__add__
#   1 of  11 in __main__.Mvar.__pow__
#   1 of   2 in __main__.Mvar.__rdiv__
#***Test Failed*** 4 failures.
