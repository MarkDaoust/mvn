"""
************************
verious helper functions
************************
"""

import itertools
import copy
import operator

import numpy

#from scipy import sqrt()
def sqrt(data):
    """
    like scipy.sqrt without a scipy depandancy

    >>> assert sqrt([1,2,3]).dtype == float
    >>> assert sqrt([1,-2,3]).dtype == complex
    """
    data = numpy.asarray(data)
    if numpy.isreal(data).all() and (data>=0).all():
        return numpy.sqrt(data)
    return data**(0.5+0j)

def mag2(vectors,axis=-1):
    """
    sum or squares along an axis
    
    >>> assert (mag2([[1,2,3],[4,5,6]]) == [14,77]);
    
    >>> assert (mag2(numpy.ones([5,10]),0) == 5*numpy.ones(10))
    >>> assert (mag2(numpy.ones([5,10]),1) == 10*numpy.ones(5))
    
    >>> assert mag2(1+1j) == 2
    """
    
    vectors = numpy.asarray(vectors)
    return numpy.real_if_close(
        (vectors*vectors.conjugate()).sum(axis)
    )

def sign(self):
    """
    improved sign function:
        returns a similar array of unit length (possibly complex) numbers pointing in the same 
        direction as the input
 
    >>> assert sign(0) == 0       
    >>> assert sign(0+0j) == 0       

    >>> assert sign(2) == 1
    >>> assert sign(-2) == 1
 
    >>> assert sign(2j) == 1j
    >>> assert sign(-2j) == -1j
    
    >>> assert sign(1+2j) == 1/sqrt(5) + 1j*2/sqrt(5)
    >>> assert sign(1-2j) == 1/sqrt(5) - 1j*2/sqrt(5)

    >>> # there's a strong connection btween `unit` and `sign`
    >>> R = numpy.random.randn([10,10])
    >>> assert (sign(R) == numpy.sign(R)).all()
    >>>
    >>> R1 = numpy.random.randn([10,10])
    >>> R2 = numpy.random.randn([10,10])
    >>>
    >>> Rj = R1+1j*R2
    >>> Rj = sign(Rj)
    >>> Rj = numpy.concatenate(Rj.real,Rj.imag,2)
    >>>
    >>> Ru = numpy.concatenate([R1,R2],2)
    >>> Ru = unit(Ru)
    >>> 
    >>> assert numpy.allclose(Rj,Ru)
    
    """
    zeros = (self == 0)
    
    self = numpy.divide(self,numpy.abs(self))
    
    self[zeros] = 0;
    
    return self
 
def unit(self,axis=-1):
    """
    along a given axis, make vectors unit length
    
    >>> assert unit([1,2,3])
    
    >>> R = numpy.random.randn([3,4,5])
    >>> axis = numpy.random.randint(0,3)
    >>> M = mag2(unit(R,axis),axis)
    >>> assert numpy.allclose(M,numpy.ones_like(M))
    
    """
    self = numpy.asarray(self)

    if axis == -1:
        axis = self.ndim-1
    
    mag = mag2(self,axis)**(0.5)

    mag = mag.reshape(mag.shape[:axis]+(1,)+mag.shape[axis:])

    return self/mag

def ascomplex(self):
    """
    return an array pointing to the same data, but interpreting it as a 
    different type
    
    >>> assert ascomplex([1,1]) == 1+1j
    """
    self = numpy.asarray(self)
    shape=self.shape
    duplicate=copy.copy(self)
    duplicate.dtype=complex
    duplicate.shape=shape[:-1]
    return duplicate

def diagstack(arrays):
    """
    stack matrixes diagonally
    1d arrays are interpreted as 1xN
    it's like numpy.diag but works with matrixes

    output is two dimensional

    type matches first input, if it is a numpy array or matrix, 
    otherwise this returns a numpy array
    """
    #make a nested matrix, with the inputs on the diagonal, and the numpy.zeros
    #function everywhere else
    E=numpy.eye(len(arrays),dtype=bool)
    result=numpy.zeros(E.shape,dtype=object)
    for index,array in zip(numpy.argwhere(E),arrays):
        index=tuple(index)
        result[index]=array
    
    result[~E]=numpy.zeros    
    
    return stack(result)


_vectorizedShape = numpy.vectorize(lambda x:x.shape)

def autoshape(rows,default=0):
    """
    >>> A = autoshape([
    ...     [    [1,2,3],   numpy.ones],
    ...     [numpy.zeros,[[1],[2],[3]]],
    ... ]) 
    >>> assert numpy.vectorize(lambda x:x.size)(A) == array([[3, 1],[9, 3]])
    """
    #first make sure everything is a 2 dimensional array
    data = [
        [[numpy.array(item,ndmin=2),callable(item)] for item in row]
        for row in rows      
    ]

    #make a 2d numpy array out of the data blocks
    data = numpy.array([
        list(itertools.chain([[None,None]],row)) 
        for row in data
    ],dtype = object
    )[:,1:,:]

    rows = data[...,0]
    calls = numpy.asarray(data[...,1],dtype = bool)

    #if anything is callable
    if calls.any():
        #make an array of shapes

        [heights,widths] = _vectorizedShape(rows)

        heights[calls]= -1
        widths[calls]= -1

        maxheight=heights.max(1)
        maxwidth=widths.max(0)
        
        #replace the -1 with the default value
        maxheight[maxheight==-1] = default
        maxwidth[maxwidth==-1] = default

        for (down,right) in numpy.argwhere(calls):
            #call the callable with (height,width) as the only argument
            #and drop it into the data slot,
            rows[down,right] = numpy.matrix(rows[down,right][0,0](
                (maxheight[down],maxwidth[right])
            ))
        
    return rows


def stack(rows,deafult = 0):
    """
    simplify matrix stacking
    vertically stack the results of horizontally stacking each row in rows, 
    with no need to explicitly declare the size of and callables
    
    >>> stack([ 
    ...     [numpy.eye(3),numpy.zeros],
    ...     [  numpy.ones,          1],
    ... ])
    matrix([[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 1.,  1.,  1.,  1.]])
    
    Callables are called with a shape tuple as the only argument. 
    The 'default' parameter controls the size when a row of column contains 
    only callables, 

    >>> stack([
    ...     [[1,2,3]],
    ...     [numpy.ones]
    ... ],default=0)
    matrix([[ 1.,  2.,  3.]])
    
    >>> stack([
    ...     [[1,2,3]],
    ...     [numpy.ones]
    ... ],default=4)
    matrix([[ 1.,  2.,  3.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> stack([
    ...     [                       [1,2,3],   1],
    ...     [lambda shape:numpy.eye(*shape),[[1]
    ...                                    , [1]]]
    ... ])
    matrix([[ 1.,  2.,  3.,  1.],
            [ 1.,  0.,  0.,  1.],
            [ 0.,  1.,  0.,  1.]])
    """
    #do the stacking    
    rows = autoshape(rows,default = 0)
    return numpy.vstack([
        numpy.hstack(row) 
        for row in rows
    ])

def parallel(*items):
    """
    resistors in parallel, and thanks to 
    duck typing and operator overloading, this happens 
    to be exactly what we need for kalman sensor fusion. 
    
    >>> assert parallel(1.0,2.0) == 1/(1/1.0+1/2.0)
    """
    inverted=[item**(-1) for item in items]
    return sum(inverted[1:],inverted[0])**(-1)

def approx(self,other = None,rtol=1e-5,atol=1e-8):
    """
    element-wise version of :py:func:`numpy.allclose`
    
    >>> assert approx(1e-12*numpy.random.randn(10,10),numpy.zeros(10,10)).all()
    
    >>> ones = numpy.ones([3,3])
    >>> eye = numpy.eye(3)+1e-6*numpy.random.randn([3,3]))
    >>> assert approx(ones,eye) == numpy.eye(3)
    """
    
    if other is None:
        other = numpy.zeros

    if callable(other):
        other=other(self.shape)
            
    delta = numpy.abs(self-other)
        
    if not delta.size:
        return numpy.empty(delta.shape,delta.dtype)
        
    infs = numpy.multiply(~numpy.isfinite(self),~numpy.isfinite(other))
    
    tol = atol + rtol*abs(other)
    
    result = delta < tol

    if infs.any():
        result[infs] = self[infs] == other[infs]
        
    return result

def dots(*args):
    """
    like numpy.dot but takes any number or arguments
    """
    assert len(args)>1
    return reduce(numpy.dot,args)

def sortrows(data,column=0):
    return data[numpy.argsort(data[:,column].flatten()),:] if data.size else data

def rotation2d(angle):
    return numpy.array([
        [ numpy.cos(angle),numpy.sin(angle)],
        [-numpy.sin(angle),numpy.cos(angle)],
    ])


def binindex(index,size):
    """
    :param index:
    :param size:
        
    convert an index to binary so it can be easily inverted
    """
    if hasattr(index,'dtype') and index.dtype==bool:
        return index
    
    binindex=numpy.zeros(size,dtype=bool)
    binindex[index]=True

    return binindex