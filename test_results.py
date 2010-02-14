from numpy import array,matrix
from mvar import Mvar
#test objects used
A= Mvar.from_attr(
    mean=
        array([  3.20227425, -10.54404517]),
    scale=
        array([[ 14.11963842,   0.        ],
               [  0.        ,  15.30468674]]),
    vectors=
        array([[-0.96369558, -0.26700341],
               [-0.26700341,  0.96369558]]),
)
B= Mvar.from_attr(
    mean=
        array([-16.46993385,   4.31430073]),
    scale=
        array([[ 23.40381419,   0.        ],
               [  0.        ,  26.08120841]]),
    vectors=
        array([[-0.09446604, -0.99552808],
               [-0.99552808,  0.09446604]]),
)
C= Mvar.from_attr(
    mean=
        array([-4.15579892,  2.29017216]),
    scale=
        array([[  1.33335905,   0.        ],
               [  0.        ,  23.48514511]]),
    vectors=
        array([[-0.550484  , -0.83484571],
               [-0.83484571,  0.550484  ]]),
)
M= matrix([[-0.07741204, -0.48682799],
        [ 0.64246847,  0.49138681]])
K1= (5.30859286432+0j)
K2= (7.19632224382+0j)
N= 5
#**********************************************************************
#File "./mvar.py", line 457, in __main__.Mvar.__and__
#Failed example:
#    assert A & B == wiki(A,B)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__and__[6]>", line 1, in <module>
#        assert A & B == wiki(A,B)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 662, in __main__.Mvar.__mul__
#Failed example:
#    assert numpy.allclose((A*numpy.eye(A.mean.size)*K1).vectors,A.vectors*K1)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[20]>", line 1, in <module>
#        assert numpy.allclose((A*numpy.eye(A.mean.size)*K1).vectors,A.vectors*K1)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 493, in __main__.Mvar.__pow__
#Failed example:
#    assert A**K1/A**K2==A**(K1-K2)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[2]>", line 1, in <module>
#        assert A**K1/A**K2==A**(K1-K2)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 503, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 789, in __main__.Mvar.__rdiv__
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
#File "./mvar.py", line 744, in __main__.Mvar.__rmul__
#Failed example:
#    assert B*B == A**2
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[4]>", line 1, in <module>
#        assert B*B == A**2
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 745, in __main__.Mvar.__rmul__
#Failed example:
#    assert A*B == A*(B.rotation.T*B.vectors) 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[5]>", line 1, in <module>
#        assert A*B == A*(B.rotation.T*B.vectors)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 754, in __main__.Mvar.__rmul__
#Failed example:
#    assert numpy.allclose(M*A, M*(A.rotation.T*A.scale*A.rotation))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[6]>", line 1, in <module>
#        assert numpy.allclose(M*A, M*(A.rotation.T*A.scale*A.rotation))
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 857, in __main__.Mvar.__sub__
#Failed example:
#    assert (A-B)+B == A
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__sub__[0]>", line 1, in <module>
#        assert (A-B)+B == A
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 859, in __main__.Mvar.__sub__
#Failed example:
#    assert numpy.allclose((A-B).cov, A.cov - B.cov)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__sub__[2]>", line 1, in <module>
#        assert numpy.allclose((A-B).cov, A.cov - B.cov)
#    AssertionError
#**********************************************************************
#6 items had failures:
#   1 of   7 in __main__.Mvar.__and__
#   1 of  24 in __main__.Mvar.__mul__
#   2 of  11 in __main__.Mvar.__pow__
#   1 of   2 in __main__.Mvar.__rdiv__
#   3 of   8 in __main__.Mvar.__rmul__
#   2 of   6 in __main__.Mvar.__sub__
#***Test Failed*** 10 failures.
