from numpy import array,matrix
from mvar import mvar
#test objects used
A= Mvar.from_attr(
    mean=
        array([ 13.68776146, -12.67890098]),
    scale=
        array([[  4.48077108+0.j,   0.00000000+0.j],
               [  0.00000000+0.j,  13.42496386+0.j]]),
    vectors=
        array([[-0.95078517+0.j,  0.30985088+0.j],
               [ 0.30985088+0.j,  0.95078517+0.j]]),
)
B= Mvar.from_attr(
    mean=
        array([-18.49182241,   3.94541866]),
    scale=
        array([[  1.93840876,   0.        ],
               [  0.        ,  16.48001545]]),
    vectors=
        array([[ 0.18102183, -0.98347908],
               [-0.98347908, -0.18102183]]),
)
C= Mvar.from_attr(
    mean=
        array([ 4.68479486,  4.60914432]),
    scale=
        array([[ 22.24274018,   0.        ],
               [  0.        ,  26.76826535]]),
    vectors=
        array([[-0.90035058, -0.4351653 ],
               [-0.4351653 ,  0.90035058]]),
)
M= matrix([[-1.49402626,  1.6294958 ],
        [ 0.48911613,  0.1063297 ]])
K1= (0.538588132519+0j)
K2= (0.712803664251+0j)
N= 7
#**********************************************************************
#File "./mvar.py", line 497, in __main__.Mvar.__pow__
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
#File "./mvar.py", line 755, in __main__.Mvar.__rmul__
#Failed example:
#    assert numpy.allclose(M/A,M*(A**(-1)))
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.__rmul__[12]>", line 1, in <module>
#        assert numpy.allclose(M/A,M*(A**(-1)))
#      File "/usr/lib/python2.5/site-packages/numpy/core/numeric.py", line 717, in allclose
#        xinf = isinf(x)
#    TypeError: function not supported for these types, and can't coerce safely to supported types
#**********************************************************************
#File "./mvar.py", line 266, in __main__.Mvar.from_data
#Failed example:
#    assert Mvar.from_data(A)==A 
#Exception raised:
#    Traceback (most recent call last):
#      File "/usr/lib/python2.5/doctest.py", line 1228, in __run
#        compileflags, 1) in test.globs
#      File "<doctest __main__.Mvar.from_data[0]>", line 1, in <module>
#        assert Mvar.from_data(A)==A
#      File "./mvar.py", line 284, in from_data
#        cov = numpy.cov(data,bias=bias,rowvar=0),
#      File "/usr/lib/python2.5/site-packages/numpy/lib/function_base.py", line 1169, in cov
#        X = array(m, ndmin=2, dtype=float)
#    TypeError: float() argument must be a string or a number
#**********************************************************************
#3 items had failures:
#   1 of  11 in __main__.Mvar.__pow__
#   1 of  13 in __main__.Mvar.__rmul__
#   1 of   1 in __main__.Mvar.from_data
#***Test Failed*** 3 failures.
