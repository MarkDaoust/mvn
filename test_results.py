from numpy import array,matrix
from mvar import Mvar
#test objects used
A= Mvar.from_attr(
    mean=
        matrix([[-16.09776668,   8.66459866]]),
    scale=
        matrix([[  7.71569616+0.j,   0.00000000+0.j],
                [  0.00000000+0.j,  21.42723491+0.j]]),
    vectors=
        matrix([[-0.88954794+0.j, -0.45684184+0.j],
                [-0.45684184+0.j,  0.88954794+0.j]]),
)
B= Mvar.from_attr(
    mean=
        matrix([[-8.14283202,  8.66013359]]),
    scale=
        matrix([[  4.95497863,   0.        ],
                [  0.        ,  13.60821337]]),
    vectors=
        matrix([[-0.68901492, -0.72474715],
                [-0.72474715,  0.68901492]]),
)
C= Mvar.from_attr(
    mean=
        matrix([[ 0.79177367, -0.04782308]]),
    scale=
        matrix([[  1.29987894,   0.        ],
                [  0.        ,  12.78439244]]),
    vectors=
        matrix([[-0.13308723, -0.99110433],
                [-0.99110433,  0.13308723]]),
)
M= matrix([[-0.01348639,  0.01340767],
        [-0.69073157,  0.32247263]])
K1= (0.551007293959+0j)
K2= (0.212371632611+0j)
N= 5
#**********************************************************************
#File "./mvar.py", line 815, in __main__.Mvar.__add__
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
#File "./mvar.py", line 817, in __main__.Mvar.__add__
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
#File "./mvar.py", line 826, in __main__.Mvar.__add__
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
#File "./mvar.py", line 457, in __main__.Mvar.__and__
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
#File "./mvar.py", line 458, in __main__.Mvar.__and__
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
#File "./mvar.py", line 462, in __main__.Mvar.__and__
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
#File "./mvar.py", line 463, in __main__.Mvar.__and__
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
#File "./mvar.py", line 469, in __main__.Mvar.__and__
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
#File "./mvar.py", line 765, in __main__.Mvar.__div__
#Failed example:
#    assert A/B == A*(B**(-1))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__div__[0]>", line 1, in <module>
#        assert A/B == A*(B**(-1))
#      File "./mvar.py", line 769, in __div__
#        return self*(other**(-1))
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 581, in __main__.Mvar.__mul__
#Failed example:
#    assert isinstance(A*B,Mvar)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[0]>", line 1, in <module>
#        assert isinstance(A*B,Mvar)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 610, in __main__.Mvar.__mul__
#Failed example:
#    assert A*A==A**2
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[8]>", line 1, in <module>
#        assert A*A==A**2
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 611, in __main__.Mvar.__mul__
#Failed example:
#    assert numpy.allclose(
#       (A*B).affine,
#       (A*dot(B.rotation.H,B.vectors)).affine
#    )
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[9]>", line 2, in <module>
#        (A*B).affine,
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 615, in __main__.Mvar.__mul__
#Failed example:
#    assert numpy.allclose(
#        (A*B).mean,dot(A.mean,B.rotation.H,B.scale,B.rotation)
#    )
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[10]>", line 2, in <module>
#        (A*B).mean,dot(A.mean,B.rotation.H,B.scale,B.rotation)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 618, in __main__.Mvar.__mul__
#Failed example:
#    assert A*(B**2) == A*(B.cov)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__mul__[11]>", line 1, in <module>
#        assert A*(B**2) == A*(B.cov)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 497, in __main__.Mvar.__pow__
#Failed example:
#    assert A**0== A**(-1)*A== A*A**(-1)== A/A        
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[0]>", line 1, in <module>
#        assert A**0== A**(-1)*A== A*A**(-1)== A/A
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 498, in __main__.Mvar.__pow__
#Failed example:
#    assert (A**K1)*(A**K2)==A**(K1+K2)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[1]>", line 1, in <module>
#        assert (A**K1)*(A**K2)==A**(K1+K2)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 499, in __main__.Mvar.__pow__
#Failed example:
#    assert A**K1/A**K2==A**(K1-K2)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[2]>", line 1, in <module>
#        assert A**K1/A**K2==A**(K1-K2)
#      File "./mvar.py", line 769, in __div__
#        return self*(other**(-1))
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 507, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 508, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 509, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 518, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 519, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 520, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 527, in __main__.Mvar.__pow__
#Failed example:
#    assert A*B==A*dot(B.rotation.H,B.scale,B.rotation)
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__pow__[10]>", line 1, in <module>
#        assert A*B==A*dot(B.rotation.H,B.scale,B.rotation)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 728, in __main__.Mvar.__rmul__
#Failed example:
#    assert A*A==A**2
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[3]>", line 1, in <module>
#        assert A*A==A**2
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 740, in __main__.Mvar.__rmul__
#Failed example:
#    assert B*B == B**2
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[4]>", line 1, in <module>
#        assert B*B == B**2
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 741, in __main__.Mvar.__rmul__
#Failed example:
#    assert A*B == A*dot(B.rotation.H,B.vectors) 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[5]>", line 1, in <module>
#        assert A*B == A*dot(B.rotation.H,B.vectors)
#      File "./mvar.py", line 673, in __mul__
#        return multipliers[type(other)](self,other)
#    KeyError: <class 'numpy.core.defmatrix.matrix'>
#**********************************************************************
#File "./mvar.py", line 278, in __main__.Mvar.from_data
#Failed example:
#    assert Mvar.from_data(A)==A 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.from_data[0]>", line 1, in <module>
#        assert Mvar.from_data(A)==A
#      File "./mvar.py", line 296, in from_data
#        cov = numpy.cov(data,bias=bias,rowvar=0),
#      File "/usr/lib/python2.5/site-packages/numpy/lib/function_base.py", line 1169, in cov
#        X = array(m, ndmin=2, dtype=float)
#    TypeError: float() argument must be a string or a number
#**********************************************************************
#7 items had failures:
#   3 of  12 in __main__.Mvar.__add__
#   5 of   7 in __main__.Mvar.__and__
#   1 of   3 in __main__.Mvar.__div__
#   5 of  24 in __main__.Mvar.__mul__
#  10 of  11 in __main__.Mvar.__pow__
#   3 of   8 in __main__.Mvar.__rmul__
#   1 of   1 in __main__.Mvar.from_data
#***Test Failed*** 28 failures.
