Traceback (most recent call last):
  File "./mvar.py", line 964, in <module>
    A=Mvar.from_attr(mean=10*numpy.random.randn(1,2),vectors=10*numpy.random.randn(2,2))
  File "./mvar.py", line 249, in from_attr
    [  1.0,   mean],
  File "./mvar.py", line 167, in __init__
    self.do_square()
  File "./mvar.py", line 183, in do_square
    if not numpy.allclose(dot(V.H,V),numpy.eye(V.shape[0]),**kwargs):
AttributeError: 'numpy.ndarray' object has no attribute 'H'
from numpy import array,matrix
from mvar import Mvar
