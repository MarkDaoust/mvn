from mvar import Mvar
from numpy import array,matrix
#test objects used
A= Mvar.from_attr(
    mean=
        matrix([[-1.01422206, -0.05026823]]),
    scale=
        matrix([[  9.52278763+0.j,   0.00000000+0.j],
                [  0.00000000+0.j,  19.59302228+0.j]]),
    vectors=
        matrix([[-0.86054321+0.j,  0.50937744+0.j],
                [ 0.50937744+0.j,  0.86054321+0.j]]),
)
B= Mvar.from_attr(
    mean=
        matrix([[  5.14284641,  22.86138346]]),
    scale=
        matrix([[ 2.55872328,  0.        ],
                [ 0.        ,  7.31122714]]),
    vectors=
        matrix([[-0.46588577, -0.88484488],
                [-0.88484488,  0.46588577]]),
)
C= Mvar.from_attr(
    mean=
        matrix([[-0.27723338, -0.1017875 ]]),
    scale=
        matrix([[ 3.57285825,  0.        ],
                [ 0.        ,  4.82499442]]),
    vectors=
        matrix([[-0.99599902,  0.0893642 ],
                [ 0.0893642 ,  0.99599902]]),
)
M= matrix([[ 0.39973162, -1.83616356],
        [ 1.05455799,  0.91061803]])
K1= (0.863788318295+0j)
K2= (0.583784152193+0j)
N= 6
#**********************************************************************
#File "./mvar.py", line 794, in __main__.Mvar.__add__
#Failed example:
#    assert (A-B)+B == A
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__add__[6]>", line 1, in <module>
#        assert (A-B)+B == A
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 796, in __main__.Mvar.__add__
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
#File "./mvar.py", line 805, in __main__.Mvar.__add__
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
#File "./mvar.py", line 446, in __main__.Mvar.__and__
#Failed example:
#    assert A & B == B & A 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__and__[0]>", line 1, in <module>
#        assert A & B == B & A
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 447, in __main__.Mvar.__and__
#Failed example:
#    assert A & B == 1/(1/A+1/B)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__and__[1]>", line 1, in <module>
#        assert A & B == 1/(1/A+1/B)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 451, in __main__.Mvar.__and__
#Failed example:
#    assert A & B & C == paralell(*abc)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__and__[4]>", line 1, in <module>
#        assert A & B & C == paralell(*abc)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 452, in __main__.Mvar.__and__
#Failed example:
#    assert A & B & C == Mvar.blend(*abc)== Mvar.__and__(*abc)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__and__[5]>", line 1, in <module>
#        assert A & B & C == Mvar.blend(*abc)== Mvar.__and__(*abc)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 458, in __main__.Mvar.__and__
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
#File "./mvar.py", line 665, in __main__.Mvar.__mul__
#Failed example:
#    assert A/B == A*(B**(-1))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[24]>", line 1, in <module>
#        assert A/B == A*(B**(-1))
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 672, in __main__.Mvar.__mul__
#Failed example:
#    assert numpy.allclose(M/A,M*(A**(-1)))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[28]>", line 1, in <module>
#        assert numpy.allclose(M/A,M*(A**(-1)))
#      File "/usr/lib/python2.5/site-packages/numpy/core/numeric.py", line 717, in allclose
#        xinf = isinf(x)
#    TypeError: function not supported for these types, and can't coerce safely to supported types
#**********************************************************************
#File "./mvar.py", line 486, in __main__.Mvar.__pow__
#Failed example:
#    assert A**0== A**(-1)*A== A*A**(-1)== A/A        
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[0]>", line 1, in <module>
#        assert A**0== A**(-1)*A== A*A**(-1)== A/A
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 496, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose((A**0).scale,numpy.eye(A.mean.size))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[3]>", line 1, in <module>
#        assert numpy.allclose((A**0).scale,numpy.eye(A.mean.size))
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 497, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose((A**0).rotation, A.rotation)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[4]>", line 1, in <module>
#        assert numpy.allclose((A**0).rotation, A.rotation)
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 498, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose(
#        (A**0).mean,
#        dot(
#            A.mean,A.rotation.H,A.scale**-1,A.rotation
#    ))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[5]>", line 4, in <module>
#        A.mean,A.rotation.H,A.scale**-1,A.rotation
#    AssertionError
#**********************************************************************
#File "./mvar.py", line 507, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose((A**K1).vectors, dot((A.scale**K1),A.rotation))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[7]>", line 1, in <module>
#        assert numpy.allclose((A**K1).vectors, dot((A.scale**K1),A.rotation))
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 276, in __pow__
#        return matrix_power(self, other)
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 94, in matrix_power
#        raise TypeError("exponent must be an integer")
#    TypeError: exponent must be an integer
#**********************************************************************
#File "./mvar.py", line 508, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose((A**K1).vectors, dot(A.vectors,A.rotation.H,A.scale**(K1-1),A.rotation))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[8]>", line 1, in <module>
#        assert numpy.allclose((A**K1).vectors, dot(A.vectors,A.rotation.H,A.scale**(K1-1),A.rotation))
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 276, in __pow__
#        return matrix_power(self, other)
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 94, in matrix_power
#        raise TypeError("exponent must be an integer")
#    TypeError: exponent must be an integer
#**********************************************************************
#File "./mvar.py", line 509, in __main__.Mvar.__pow__
#Failed example:
#    assert numpy.allclose((A**K1).mean,dot(A.mean,A.rotation.H,A.scale**(K1-1),A.rotation))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[9]>", line 1, in <module>
#        assert numpy.allclose((A**K1).mean,dot(A.mean,A.rotation.H,A.scale**(K1-1),A.rotation))
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 276, in __pow__
#        return matrix_power(self, other)
#      File "/usr/lib/python2.5/site-packages/numpy/core/defmatrix.py", line 94, in matrix_power
#        raise TypeError("exponent must be an integer")
#    TypeError: exponent must be an integer
#**********************************************************************
#File "./mvar.py", line 267, in __main__.Mvar.from_data
#Failed example:
#    assert Mvar.from_data(A)==A 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.from_data[0]>", line 1, in <module>
#        assert Mvar.from_data(A)==A
#      File "./mvar.py", line 285, in from_data
#        cov = numpy.cov(data,bias=bias,rowvar=0),
#      File "/usr/lib/python2.5/site-packages/numpy/lib/function_base.py", line 1169, in cov
#        X = array(m, ndmin=2, dtype=float)
#    TypeError: float() argument must be a string or a number
#**********************************************************************
#5 items had failures:
#   3 of  12 in __main__.Mvar.__add__
#   5 of   7 in __main__.Mvar.__and__
#   2 of  29 in __main__.Mvar.__mul__
#   7 of  11 in __main__.Mvar.__pow__
#   1 of   1 in __main__.Mvar.from_data
#***Test Failed*** 18 failures.
